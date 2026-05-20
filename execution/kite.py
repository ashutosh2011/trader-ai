"""Live Kite Connect broker."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

import structlog

from config.settings import AppSettings
from core.signal import Signal
from execution.broker import (
    Broker,
    OrderResult,
    Position,
    deterministic_client_order_id,
)
from risk.manager import is_kill_switch_active

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

PositionSide = Literal["LONG", "SHORT"]


class KiteBrokerClient(Protocol):
    """Kite API surface required by :class:`KiteBroker`."""

    def place_order(self, **kwargs: Any) -> str: ...

    def orders(self) -> list[dict[str, Any]]: ...

    def positions(self) -> dict[str, list[dict[str, Any]]]: ...


class KiteBroker(Broker):
    """Live broker via Kite Connect bracket (BO) orders.

    TRADEOFF: Uses BO variety with absolute SL/target prices from the signal;
    exchange min tick/lot constraints are not re-validated here beyond qty > 0.
    """

    def __init__(
        self,
        client: KiteBrokerClient,
        *,
        settings: AppSettings | None = None,
        exchange: str = "NSE",
        product: str = "MIS",
    ) -> None:
        self._client = client
        self._settings = settings or AppSettings()
        self._exchange = exchange
        self._product = product
        self._placed_ids: set[str] = set()

    def place_bracket_order(self, signal: Signal, qty: int) -> OrderResult:
        """Place an idempotent bracket order; rejects on kill switch or bad qty."""
        order_id = deterministic_client_order_id(
            signal.strategy_id, signal.ts, signal.symbol
        )
        if is_kill_switch_active(
            kill_file=self._settings.kill_switch_file,
            env_var=self._settings.kill_switch_env,
        ):
            return _reject(signal, qty, order_id, "kill_switch_active")
        if qty < 1:
            return _reject(signal, qty, order_id, "qty must be >= 1")
        if order_id in self._placed_ids:
            logger.info("kite_order_idempotent_skip", order_id=order_id)
            return OrderResult(
                client_order_id=order_id,
                status="PENDING",
                symbol=signal.symbol,
                side=signal.side,
                qty=qty,
                fill_price=0.0,
                message="duplicate_client_order_id",
            )

        transaction_type = "BUY" if signal.side == "BUY" else "SELL"
        try:
            broker_order_id = self._client.place_order(
                variety="bo",
                exchange=self._exchange,
                tradingsymbol=signal.symbol,
                transaction_type=transaction_type,
                quantity=qty,
                product=self._product,
                order_type="LIMIT",
                price=signal.entry,
                validity="DAY",
                squareoff=abs(signal.target - signal.entry),
                stoploss=abs(signal.entry - signal.stop_loss),
                tag=order_id,
            )
            self._placed_ids.add(order_id)
            logger.info(
                "kite_bracket_placed",
                client_order_id=order_id,
                broker_order_id=broker_order_id,
                symbol=signal.symbol,
            )
            return OrderResult(
                client_order_id=order_id,
                status="PENDING",
                symbol=signal.symbol,
                side=signal.side,
                qty=qty,
                fill_price=signal.entry,
                message=f"broker_order_id={broker_order_id}",
            )
        except Exception as exc:
            logger.exception("kite_order_failed", symbol=signal.symbol, error=str(exc))
            return _reject(signal, qty, order_id, str(exc))

    def get_positions(self) -> list[Position]:
        """Map Kite net positions to :class:`Position` models."""
        raw = self._client.positions()
        net = raw.get("net", [])
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
            positions.append(
                Position(
                    symbol=symbol,
                    side=side,
                    qty=abs(qty),
                    entry_price=entry,
                    stop_loss=entry,
                    target=entry,
                    strategy_id="kite_sync",
                    opened_at=_now_ist(),
                )
            )
        return positions

    def flatten_all(self) -> None:
        """Market-close all open net positions."""
        for position in self.get_positions():
            side = "SELL" if position.side == "LONG" else "BUY"
            try:
                self._client.place_order(
                    variety="regular",
                    exchange=self._exchange,
                    tradingsymbol=position.symbol,
                    transaction_type=side,
                    quantity=position.qty,
                    product=self._product,
                    order_type="MARKET",
                    validity="DAY",
                )
            except Exception:
                logger.exception("kite_flatten_failed", symbol=position.symbol)


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
    from datetime import datetime

    return datetime.now(tz=IST)
