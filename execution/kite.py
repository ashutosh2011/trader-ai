"""Live Kite Connect broker.

Architecture: entry-as-LIMIT plus a single OCO (two-leg) GTT carrying the
SL trigger and the target trigger. Polling-driven state machine, persisted
to DuckDB via :class:`OrderStateStore`.

TRADEOFF (BO replacement): Zerodha disabled bracket orders (variety="bo") in
March 2020. We replicate the bracket semantics with: (entry LIMIT order)
+ (one OCO GTT once the entry fills). The OCO trigger guarantees that when
either SL or target fires, the other side is auto-cancelled by the broker —
this is functionally identical to a BO's child-order cancel-on-fill.

TRADEOFF (OCO over two single-leg GTTs): a single two-leg GTT removes one
class of races (we never have to cancel the opposite leg ourselves on a
fill) and halves the GTT count which matters because Kite caps active GTTs
per account. The legacy two-GTT design is documented in case Kite ever
removes OCO support.

TRADEOFF (polling over WebSocket postbacks): The orchestrator already runs
at 1m bar cadence; polling ``client.orders()`` + ``client.get_gtts()`` once
per bar yields deterministic, replayable state transitions and is trivial to
debug. A WS-postback hook can be added later; the state machine and store
will absorb it without changes.

TRADEOFF (market-on-trigger SL): SL leg fires as a MARKET order rather than
SL-LIMIT to guarantee the exit fills in fast-moving markets; the slippage
cost on a stop is acceptable for personal-account size, and a stuck
SL-LIMIT is the worse failure mode.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

import structlog
from kiteconnect import exceptions as kite_exceptions

from config.settings import AppSettings
from core.signal import Signal
from execution.broker import (
    Broker,
    FlattenIncomplete,
    OrderResult,
    Position,
    deterministic_client_order_id,
)
from execution.order_state import (
    OPEN_STATES,
    OrderRecord,
    OrderState,
    OrderStateStore,
)
from execution.retry import with_retry
from risk.manager import is_kill_switch_active

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

PositionSide = Literal["LONG", "SHORT"]

# Exception types that signal a transient broker/network problem and are
# worth a retry. ``InputException`` and ``OrderException`` are explicitly
# omitted — they indicate malformed payloads or rejected orders, and
# retrying them is at best wasted work and at worst a double-submit.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    kite_exceptions.NetworkException,
    kite_exceptions.GeneralException,
    kite_exceptions.DataException,
)

_FLATTEN_POLL_BUDGET = 10
_FLATTEN_POLL_INTERVAL_SEC = 1.0

_TERMINAL_ENTRY_REJECTED = {"REJECTED", "CANCELLED"}
_TERMINAL_ENTRY_FILLED = "COMPLETE"


class KiteBrokerClient(Protocol):
    """Subset of the Kite Connect surface required by :class:`KiteBroker`."""

    def place_order(self, **kwargs: Any) -> str: ...

    def place_gtt(self, **kwargs: Any) -> dict[str, Any]: ...

    def get_gtts(self) -> list[dict[str, Any]]: ...

    def delete_gtt(self, trigger_id: int) -> dict[str, Any]: ...

    def orders(self) -> list[dict[str, Any]]: ...

    def positions(self) -> dict[str, list[dict[str, Any]]]: ...


class KiteBroker(Broker):
    """Live broker built on Kite REST: LIMIT entry + OCO GTT exit.

    Args:
        client: Kite client (typically :class:`data.kite_client.KiteClient`,
            which already rate-limits and adds the token refresh hook).
        settings: Optional :class:`AppSettings`; defaults to a fresh instance.
        state_store: Persistent :class:`OrderStateStore` used for
            idempotency and the polling state machine. If ``None`` is passed
            the broker still operates but every restart is treated as a
            fresh session — only do this in tests.
        exchange: Default exchange used when placing orders (e.g. ``NSE``).
        product: Force a specific product (``MIS``, ``NRML``, ``CNC``).
            ``None`` (default) picks ``NRML`` for F&O symbols heuristically
            (suffix ``FUT``, ``CE``, ``PE``) and ``MIS`` otherwise.

    TRADEOFF: Product auto-detection from the trading symbol is a string
    heuristic — it covers the standard NSE F&O symbology but mis-classifies
    exotic names. Pass ``product=`` explicitly when the symbol naming is
    unusual.
    """

    def __init__(
        self,
        client: KiteBrokerClient,
        *,
        settings: AppSettings | None = None,
        state_store: OrderStateStore | None = None,
        exchange: str = "NSE",
        product: str | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._settings = settings or AppSettings()
        self._state_store = state_store
        self._exchange = exchange
        self._product = product
        self._clock = clock or _now_ist
        self._sleep = sleep
        self._memory_ids: set[str] = set()

    @property
    def state_store(self) -> OrderStateStore | None:
        """Expose the persistent state store (used by reconciler)."""
        return self._state_store

    def place_bracket_order(self, signal: Signal, qty: int) -> OrderResult:
        """Place the entry LIMIT order and persist a ``PENDING_ENTRY`` record.

        The OCO GTT is *not* placed here — it goes out only once
        :meth:`poll_and_advance` observes that the entry filled. This keeps
        a rejected entry from leaving a dangling GTT and guarantees the GTT
        carries the actual fill price as ``last_price``.
        """
        client_order_id = deterministic_client_order_id(
            signal.strategy_id, signal.ts, signal.symbol
        )
        if is_kill_switch_active(
            kill_file=self._settings.kill_switch_file,
            env_var=self._settings.kill_switch_env,
        ):
            return _reject(signal, qty, client_order_id, "kill_switch_active")
        if qty < 1:
            return _reject(signal, qty, client_order_id, "qty must be >= 1")

        existing = self._lookup_existing(client_order_id)
        if existing is not None:
            logger.info(
                "kite_order_idempotent_skip",
                client_order_id=client_order_id,
                state=existing.state.value if existing.state else "unknown",
            )
            return OrderResult(
                client_order_id=client_order_id,
                status="PENDING",
                symbol=signal.symbol,
                side=signal.side,
                qty=qty,
                fill_price=0.0,
                message="duplicate_client_order_id",
            )

        transaction_type = "BUY" if signal.side == "BUY" else "SELL"
        product = self._resolve_product(signal.symbol)
        try:
            broker_order_id = with_retry(
                lambda: self._client.place_order(
                    variety="regular",
                    exchange=self._exchange,
                    tradingsymbol=signal.symbol,
                    transaction_type=transaction_type,
                    quantity=qty,
                    product=product,
                    order_type="LIMIT",
                    price=signal.entry,
                    validity="DAY",
                    tag=client_order_id,
                ),
                retries=3,
                base_delay=0.5,
                max_delay=4.0,
                retryable=RETRYABLE_EXCEPTIONS,
                logger_=logger,
                op="place_entry",
                sleep=self._sleep,
            )
        except (kite_exceptions.InputException, kite_exceptions.OrderException) as exc:
            logger.exception(
                "kite_order_rejected_bad_input",
                symbol=signal.symbol,
                error=str(exc),
            )
            return _reject(signal, qty, client_order_id, str(exc))
        except Exception as exc:
            logger.exception("kite_order_failed", symbol=signal.symbol, error=str(exc))
            return _reject(signal, qty, client_order_id, str(exc))

        now = self._clock()
        record = OrderRecord(
            client_order_id=client_order_id,
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            entry_price=signal.entry,
            stop_loss=signal.stop_loss,
            target=signal.target,
            state=OrderState.PENDING_ENTRY,
            entry_order_id=broker_order_id,
            signal_ts=signal.ts,
            created_at=now,
            updated_at=now,
            strategy_id=signal.strategy_id,
        )
        self._persist(record)
        logger.info(
            "kite_entry_placed",
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
            symbol=signal.symbol,
        )
        return OrderResult(
            client_order_id=client_order_id,
            status="PENDING",
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            fill_price=signal.entry,
            message=f"broker_order_id={broker_order_id}",
        )

    def poll_and_advance(self) -> None:
        """Walk every open record forward using the broker's view.

        Should be called once per orchestrator tick. Catches and logs
        transient errors so a flaky network does not crash the loop.
        """
        if self._state_store is None:
            return
        open_records = self._state_store.list_open()
        if not open_records:
            return
        try:
            orders = with_retry(
                self._client.orders,
                retries=3,
                base_delay=0.5,
                max_delay=4.0,
                retryable=RETRYABLE_EXCEPTIONS,
                logger_=logger,
                op="poll_orders",
                sleep=self._sleep,
            )
        except kite_exceptions.KiteException as exc:
            logger.warning("kite_poll_orders_failed", error=str(exc))
            return

        try:
            gtts = with_retry(
                self._client.get_gtts,
                retries=3,
                base_delay=0.5,
                max_delay=4.0,
                retryable=RETRYABLE_EXCEPTIONS,
                logger_=logger,
                op="poll_gtts",
                sleep=self._sleep,
            )
        except kite_exceptions.KiteException as exc:
            logger.warning("kite_poll_gtts_failed", error=str(exc))
            gtts = []

        orders_by_id = {str(o.get("order_id")): o for o in orders}
        gtts_by_id = {int(g.get("id", 0)): g for g in gtts if g.get("id") is not None}

        for record in open_records:
            if record.state == OrderState.PENDING_ENTRY:
                self._advance_pending(record, orders_by_id)
            elif record.state == OrderState.ENTERED:
                self._advance_entered(record, orders_by_id, gtts_by_id)

    def get_positions(self) -> list[Position]:
        """Return live net positions, enriched with SL/target from state."""
        raw = self._client.positions()
        net = raw.get("net", [])
        records_by_symbol: dict[str, OrderRecord] = {}
        if self._state_store is not None:
            for record in self._state_store.list_open():
                if record.state == OrderState.ENTERED:
                    records_by_symbol[record.symbol] = record

        positions: list[Position] = []
        for row in net:
            qty = int(row.get("quantity", 0))
            if qty == 0:
                continue
            side: PositionSide = "LONG" if qty > 0 else "SHORT"
            entry = float(row.get("average_price", 0))
            if entry <= 0:
                continue
            symbol = str(row.get("tradingsymbol", ""))
            matched: OrderRecord | None = records_by_symbol.get(symbol)
            positions.append(
                Position(
                    symbol=symbol,
                    side=side,
                    qty=abs(qty),
                    entry_price=entry,
                    stop_loss=matched.stop_loss if matched is not None else None,
                    target=matched.target if matched is not None else None,
                    strategy_id=(
                        matched.strategy_id if matched is not None else "kite_sync"
                    ),
                    opened_at=self._clock(),
                )
            )
        return positions

    def flatten_all(self) -> None:
        """Square off every open net position at market, with confirmation.

        Cancels the OCO GTT for each ``ENTERED`` record, places a MARKET
        square-off order per position, then polls :meth:`get_positions`
        until everything is zero or ``_FLATTEN_POLL_BUDGET`` is exhausted.

        Raises:
            FlattenIncomplete: When positions remain after the poll budget.
        """
        # Cancel any tracked GTTs first so a fresh-triggered exit cannot
        # race the square-off and double-flatten.
        if self._state_store is not None:
            for record in self._state_store.list_open():
                if record.state != OrderState.ENTERED or record.sl_gtt_id is None:
                    continue
                self._safe_delete_gtt(record.sl_gtt_id)
                self._mark_cancelled(record)

        for position in self.get_positions():
            self._place_market_square_off(position)

        for attempt in range(1, _FLATTEN_POLL_BUDGET + 1):
            try:
                residual = self.get_positions()
            except Exception as exc:
                logger.warning("kite_flatten_poll_error", attempt=attempt, error=str(exc))
                residual = []
            if not residual:
                logger.info("kite_flatten_complete", attempts=attempt)
                return
            logger.info(
                "kite_flatten_residual",
                attempt=attempt,
                budget=_FLATTEN_POLL_BUDGET,
                open=[p.symbol for p in residual],
            )
            self._sleep(_FLATTEN_POLL_INTERVAL_SEC)

        final = self.get_positions()
        if final:
            logger.error(
                "kite_flatten_incomplete",
                attempts=_FLATTEN_POLL_BUDGET,
                open=[p.symbol for p in final],
            )
            raise FlattenIncomplete(open_positions=final, attempts=_FLATTEN_POLL_BUDGET)

    def _place_market_square_off(self, position: Position) -> None:
        """Place one MARKET square-off order for ``position`` with retry."""
        side = "SELL" if position.side == "LONG" else "BUY"
        product = self._resolve_product(position.symbol)

        def _do_place() -> str:
            return self._client.place_order(
                variety="regular",
                exchange=self._exchange,
                tradingsymbol=position.symbol,
                transaction_type=side,
                quantity=position.qty,
                product=product,
                order_type="MARKET",
                validity="DAY",
            )

        try:
            with_retry(
                _do_place,
                retries=3,
                base_delay=0.5,
                max_delay=4.0,
                retryable=RETRYABLE_EXCEPTIONS,
                logger_=logger,
                op="flatten_market",
                sleep=self._sleep,
            )
        except Exception as exc:
            logger.exception(
                "kite_flatten_market_failed",
                symbol=position.symbol,
                error=str(exc),
            )

    def _advance_pending(
        self,
        record: OrderRecord,
        orders_by_id: dict[str, dict[str, Any]],
    ) -> None:
        if record.entry_order_id is None:
            return
        order = orders_by_id.get(record.entry_order_id)
        if order is None:
            return
        status = str(order.get("status", "")).upper()
        if status == _TERMINAL_ENTRY_FILLED:
            avg_price = _safe_float(order.get("average_price"))
            fill_price = avg_price if avg_price > 0 else record.entry_price
            self._place_oco_gtt(record, fill_price)
        elif status in _TERMINAL_ENTRY_REJECTED:
            updated = record.model_copy(
                update={
                    "state": OrderState.FAILED,
                    "error": str(order.get("status_message") or status),
                    "updated_at": self._clock(),
                }
            )
            self._persist(updated)
            logger.info(
                "kite_entry_failed",
                client_order_id=record.client_order_id,
                status=status,
            )

    def _advance_entered(
        self,
        record: OrderRecord,
        orders_by_id: dict[str, dict[str, Any]],
        gtts_by_id: dict[int, dict[str, Any]],
    ) -> None:
        if record.sl_gtt_id is None:
            return
        gtt = gtts_by_id.get(record.sl_gtt_id)
        if gtt is None:
            # GTT vanished — surface as drift, leave record open for manual
            # review. The reconciler will pick this up on next startup.
            logger.warning(
                "kite_gtt_missing",
                client_order_id=record.client_order_id,
                gtt_id=record.sl_gtt_id,
            )
            return
        status = str(gtt.get("status", "")).lower()
        if status != "triggered":
            return

        exit_price = _extract_exit_price(record, gtt, orders_by_id)
        sign = 1.0 if record.side == "BUY" else -1.0
        fill_price = record.fill_price if record.fill_price is not None else record.entry_price
        pnl = (exit_price - fill_price) * record.qty * sign
        reason = _classify_exit(record, exit_price)
        updated = record.model_copy(
            update={
                "state": OrderState.EXITED,
                "exit_price": exit_price,
                "pnl": pnl,
                "updated_at": self._clock(),
            }
        )
        self._persist(updated)
        logger.info(
            "kite_position_exited",
            client_order_id=record.client_order_id,
            symbol=record.symbol,
            exit_price=exit_price,
            pnl=pnl,
            reason=reason,
        )

    def _place_oco_gtt(self, record: OrderRecord, fill_price: float) -> None:
        try:
            payload = _build_oco_gtt_payload(
                record=record,
                fill_price=fill_price,
                exchange=self._exchange,
                product=self._resolve_product(record.symbol),
            )
            response = with_retry(
                lambda: self._client.place_gtt(**payload),
                retries=3,
                base_delay=0.5,
                max_delay=4.0,
                retryable=RETRYABLE_EXCEPTIONS,
                logger_=logger,
                op="place_oco_gtt",
                sleep=self._sleep,
            )
        except (kite_exceptions.InputException, kite_exceptions.OrderException) as exc:
            self._mark_failed(record, f"gtt_rejected: {exc}")
            return
        except Exception as exc:
            self._mark_failed(record, f"gtt_error: {exc}")
            return

        trigger_id = _extract_trigger_id(response)
        if trigger_id is None:
            self._mark_failed(record, "gtt_no_trigger_id")
            return

        updated = record.model_copy(
            update={
                "state": OrderState.ENTERED,
                "sl_gtt_id": trigger_id,
                "target_gtt_id": None,
                "fill_price": fill_price,
                "updated_at": self._clock(),
            }
        )
        self._persist(updated)
        logger.info(
            "kite_entered",
            client_order_id=record.client_order_id,
            gtt_id=trigger_id,
            fill_price=fill_price,
        )

    def _lookup_existing(self, client_order_id: str) -> OrderRecord | None:
        if self._state_store is not None:
            existing = self._state_store.get(client_order_id)
            if existing is not None and existing.state in OPEN_STATES:
                return existing
            if existing is not None and existing.state == OrderState.EXITED:
                return existing
            return None
        if client_order_id in self._memory_ids:
            # Minimal in-memory fallback for tests / no-store mode.
            now = self._clock()
            return OrderRecord(
                client_order_id=client_order_id,
                symbol="",
                side="BUY",
                qty=1,
                entry_price=1.0,
                stop_loss=0.5,
                target=1.5,
                state=OrderState.PENDING_ENTRY,
                signal_ts=now,
                created_at=now,
                updated_at=now,
                strategy_id="memory_fallback",
            )
        return None

    def _persist(self, record: OrderRecord) -> None:
        if self._state_store is not None:
            self._state_store.upsert(record)
        self._memory_ids.add(record.client_order_id)

    def _mark_failed(self, record: OrderRecord, reason: str) -> None:
        updated = record.model_copy(
            update={
                "state": OrderState.FAILED,
                "error": reason,
                "updated_at": self._clock(),
            }
        )
        self._persist(updated)
        logger.warning(
            "kite_record_failed",
            client_order_id=record.client_order_id,
            reason=reason,
        )

    def _mark_cancelled(self, record: OrderRecord) -> None:
        updated = record.model_copy(
            update={
                "state": OrderState.CANCELLED,
                "updated_at": self._clock(),
            }
        )
        self._persist(updated)

    def _safe_delete_gtt(self, trigger_id: int) -> None:
        try:
            with_retry(
                lambda: self._client.delete_gtt(trigger_id),
                retries=3,
                base_delay=0.5,
                max_delay=4.0,
                retryable=RETRYABLE_EXCEPTIONS,
                logger_=logger,
                op="delete_gtt",
                sleep=self._sleep,
            )
        except Exception as exc:
            logger.warning(
                "kite_delete_gtt_failed",
                trigger_id=trigger_id,
                error=str(exc),
            )

    def _resolve_product(self, symbol: str) -> str:
        if self._product is not None:
            return self._product
        upper = symbol.upper()
        # TRADEOFF: heuristic; explicit ctor arg always wins.
        if upper.endswith(("FUT", "CE", "PE")):
            return "NRML"
        return "MIS"


def _build_oco_gtt_payload(
    *,
    record: OrderRecord,
    fill_price: float,
    exchange: str,
    product: str,
) -> dict[str, Any]:
    """Construct the ``place_gtt`` kwargs for the OCO exit pair.

    Kite expects ``trigger_values`` sorted ascending and ``orders`` aligned
    with them. For a LONG the lower trigger is the stop-loss and the upper
    is the target; for a SHORT the relationship inverts.
    """
    qty = record.qty
    if record.side == "BUY":
        # LONG: SL is below entry, target is above.
        lower_trigger = record.stop_loss
        upper_trigger = record.target
        lower_order = _gtt_market_leg(
            exchange=exchange,
            symbol=record.symbol,
            transaction_type="SELL",
            quantity=qty,
            product=product,
            price=record.stop_loss,
        )
        upper_order = _gtt_limit_leg(
            exchange=exchange,
            symbol=record.symbol,
            transaction_type="SELL",
            quantity=qty,
            product=product,
            price=record.target,
        )
    else:
        # SHORT: target is below entry, SL is above.
        lower_trigger = record.target
        upper_trigger = record.stop_loss
        lower_order = _gtt_limit_leg(
            exchange=exchange,
            symbol=record.symbol,
            transaction_type="BUY",
            quantity=qty,
            product=product,
            price=record.target,
        )
        upper_order = _gtt_market_leg(
            exchange=exchange,
            symbol=record.symbol,
            transaction_type="BUY",
            quantity=qty,
            product=product,
            price=record.stop_loss,
        )

    return {
        "trigger_type": "two-leg",
        "tradingsymbol": record.symbol,
        "exchange": exchange,
        "trigger_values": [lower_trigger, upper_trigger],
        "last_price": fill_price,
        "orders": [lower_order, upper_order],
    }


def _gtt_market_leg(
    *,
    exchange: str,
    symbol: str,
    transaction_type: str,
    quantity: int,
    product: str,
    price: float,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "tradingsymbol": symbol,
        "transaction_type": transaction_type,
        "quantity": quantity,
        "product": product,
        "order_type": "MARKET",
        # TRADEOFF: ``price`` is still required by the kiteconnect payload
        # validator even for MARKET legs — we pass the trigger as the
        # nominal price; Kite ignores it for execution. See
        # ``KiteConnect._get_gtt_payload``.
        "price": price,
    }


def _gtt_limit_leg(
    *,
    exchange: str,
    symbol: str,
    transaction_type: str,
    quantity: int,
    product: str,
    price: float,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "tradingsymbol": symbol,
        "transaction_type": transaction_type,
        "quantity": quantity,
        "product": product,
        "order_type": "LIMIT",
        "price": price,
    }


def _extract_trigger_id(response: dict[str, Any]) -> int | None:
    """Pull the GTT id from ``place_gtt``'s response.

    Kite returns ``{"trigger_id": <int>}`` in current API versions; older
    snapshots used ``{"id": <int>}``. We try both.
    """
    for key in ("trigger_id", "id"):
        value = response.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_exit_price(
    record: OrderRecord,
    gtt: dict[str, Any],
    orders_by_id: dict[str, dict[str, Any]],
) -> float:
    """Pick the best available exit fill price after a GTT triggers.

    Preference order:
        1. The triggered leg's resulting ``average_price`` from
           ``client.orders()`` if Kite has attached the order id to the
           GTT result.
        2. The triggered leg's configured price (target limit / stop value)
           — accurate for limit legs, approximate for market legs.
        3. Fall back to the stop-loss level as a conservative estimate.
    """
    triggered_price = _extract_triggered_avg_price(gtt, orders_by_id)
    if triggered_price is not None:
        return triggered_price
    triggered_value = _safe_float(gtt.get("triggered_at_price"))
    if triggered_value > 0:
        return triggered_value
    return record.stop_loss


def _extract_triggered_avg_price(
    gtt: dict[str, Any],
    orders_by_id: dict[str, dict[str, Any]],
) -> float | None:
    result = gtt.get("result")
    if not isinstance(result, dict):
        return None
    order_result = result.get("order_result")
    if isinstance(order_result, dict):
        order_id = order_result.get("order_id")
        if order_id is not None:
            row = orders_by_id.get(str(order_id))
            if row is not None:
                avg = _safe_float(row.get("average_price"))
                if avg > 0:
                    return avg
        avg = _safe_float(order_result.get("average_price"))
        if avg > 0:
            return avg
    return None


def _classify_exit(record: OrderRecord, exit_price: float) -> str:
    sl_distance = abs(exit_price - record.stop_loss)
    target_distance = abs(exit_price - record.target)
    if sl_distance <= target_distance:
        return "stop_loss"
    return "target"


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _reject(signal: Signal, qty: int, order_id: str, message: str) -> OrderResult:
    return OrderResult(
        client_order_id=order_id,
        status="REJECTED",
        symbol=signal.symbol,
        side=signal.side,
        qty=qty,
        fill_price=0.0,
        message=message,
    )


def _now_ist() -> datetime:
    return datetime.now(tz=IST)
