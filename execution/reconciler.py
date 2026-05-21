"""Restart-time broker vs local state reconciliation.

The reconciler used to extract bracket levels from the Kite orderbook's
child legs (the bracket-order design). With BO retired, the source of
truth for SL/target is now :class:`execution.order_state.OrderStateStore` —
populated by :class:`execution.kite.KiteBroker` as it walks orders through
the lifecycle. On startup we cross-check the persisted records against the
broker's live positions and surface any drift.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import structlog
from pydantic import BaseModel, ConfigDict, Field

from execution.broker import Broker, Position
from execution.order_state import OPEN_STATES, OrderRecord, OrderStateStore

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

PositionSide = Literal["LONG", "SHORT"]

# Kite order statuses considered "open" — visible in the day's orderbook
# but not yet finalized. Used purely for reporting in :class:`ReconciledOrder`.
_OPEN_ORDER_STATUSES = {"OPEN", "TRIGGER PENDING", "PUT ORDER REQ RECEIVED"}


class LocalPositionSnapshot(BaseModel):
    """Persisted local position row (legacy JSON snapshot format)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: PositionSide
    qty: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    strategy_id: str


class ReconciledOrder(BaseModel):
    """Open order observed at reconciliation."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    symbol: str
    status: str
    qty: int
    price: float


class ReconciledState(BaseModel):
    """Broker-aligned state after startup reconciliation."""

    model_config = ConfigDict(frozen=True)

    reconciled_at: datetime
    positions: list[Position]
    open_orders: list[ReconciledOrder]
    drift_symbols: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class StateReconciler:
    """Cross-check broker positions/orders against the persistent state store.

    On startup the broker may know of GTT-driven brackets that were placed
    in a previous process. We:

    * Read live broker positions.
    * Back-fill ``stop_loss`` / ``target`` onto each position from the
      matching :class:`OrderRecord` (matched by ``symbol`` and the position's
      open ``ENTERED`` record, with a fallback to ``entry_order_id`` from
      the orderbook).
    * Surface drift:

      * A broker position with no matching open record (someone traded
        outside this system).
      * An ``ENTERED`` record with no matching broker position (the
        position was closed externally — manual square-off, expiry, etc.).

    TRADEOFF: We no longer fabricate bracket levels by walking child legs;
    GTT-based brackets do not have child legs in the orderbook. Records
    that are missing from the store (e.g. legacy positions from before
    this batch) will keep ``stop_loss = target = None`` and the orchestrator
    will skip bar-level exit math for them — the broker's own GTT remains
    authoritative.
    """

    def __init__(
        self,
        broker: Broker,
        local_state_path: Path | None = None,
        *,
        state_store: OrderStateStore | None = None,
    ) -> None:
        self._broker = broker
        self._local_path = local_state_path
        self._store = state_store

    def reconcile(
        self,
        *,
        orders: list[dict[str, Any]] | None = None,
        state_store: OrderStateStore | None = None,
    ) -> ReconciledState:
        """Reconcile broker positions/orders with the persistent state store.

        Args:
            orders: Today's orderbook rows (Kite ``orders()`` shape).
                Populates ``open_orders`` for diagnostics. Not required for
                back-filling bracket levels.
            state_store: Optional store override. Falls back to the one
                passed at construction time.

        Returns:
            A :class:`ReconciledState` snapshot.
        """
        order_rows = orders or []
        store = state_store or self._store
        broker_positions = self._broker.get_positions()
        records = store.list_open() if store is not None else []
        enriched_positions = _backfill_from_records(broker_positions, records, order_rows)
        open_orders = _map_open_orders(order_rows)

        local = _load_local(self._local_path)
        drift = _detect_drift(enriched_positions, local, records)

        notes: list[str] = []
        if drift:
            notes.append(f"position_drift: {','.join(drift)}")
            logger.warning("reconcile_drift", symbols=drift)
        backfilled = sum(
            1 for p in enriched_positions
            if p.stop_loss is not None or p.target is not None
        )
        state = ReconciledState(
            reconciled_at=datetime.now(tz=IST),
            positions=enriched_positions,
            open_orders=open_orders,
            drift_symbols=drift,
            notes=notes,
        )
        logger.info(
            "reconcile_complete",
            positions=len(enriched_positions),
            open_orders=len(open_orders),
            drift=len(drift),
            backfilled=backfilled,
        )
        return state


def _backfill_from_records(
    positions: list[Position],
    records: list[OrderRecord],
    orders: list[dict[str, Any]],
) -> list[Position]:
    """Enrich broker positions with SL/target levels sourced from records.

    Match priority:

        1. Position symbol equals record symbol AND the broker orderbook
           has a row with ``tag`` == record's ``client_order_id``.
        2. Fall back to plain symbol match if exactly one record covers
           the symbol — handles the common single-strategy-per-symbol
           case where tags may not have been persisted in the orderbook.
    """
    if not records:
        return list(positions)

    tag_to_symbol: dict[str, str] = {}
    for row in orders:
        tag = str(row.get("tag", "") or "").strip()
        symbol = str(row.get("tradingsymbol", "") or "").strip()
        if tag and symbol:
            tag_to_symbol[tag] = symbol

    records_by_symbol: dict[str, list[OrderRecord]] = {}
    for record in records:
        if record.state not in OPEN_STATES:
            continue
        records_by_symbol.setdefault(record.symbol, []).append(record)

    enriched: list[Position] = []
    for position in positions:
        candidates = records_by_symbol.get(position.symbol, [])
        picked: OrderRecord | None = _pick_record(candidates, position, tag_to_symbol)
        if picked is None:
            enriched.append(position)
            continue
        enriched.append(
            position.model_copy(
                update={
                    "stop_loss": picked.stop_loss,
                    "target": picked.target,
                    "strategy_id": picked.strategy_id or position.strategy_id,
                }
            )
        )
    return enriched


def _pick_record(
    candidates: list[OrderRecord],
    position: Position,
    tag_to_symbol: dict[str, str],
) -> OrderRecord | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Multiple ENTERED records on the same symbol — prefer the one whose
    # client_order_id is present in the orderbook tags (i.e. the entry
    # order is still trackable) and matches the same tradingsymbol.
    for record in candidates:
        if tag_to_symbol.get(record.client_order_id) == position.symbol:
            return record
    # TRADEOFF: ambiguous; pick the most recently updated record. The
    # alternative — refusing to back-fill — leaves SL/target as None which
    # silently disables bar-level exit math; explicit selection at least
    # leaves a single deterministic choice and a log line.
    candidates.sort(key=lambda r: r.updated_at, reverse=True)
    logger.warning(
        "reconcile_ambiguous_records",
        symbol=position.symbol,
        candidates=[r.client_order_id for r in candidates],
    )
    return candidates[0]


def _map_open_orders(rows: list[dict[str, Any]]) -> list[ReconciledOrder]:
    result: list[ReconciledOrder] = []
    for row in rows:
        status = str(row.get("status", ""))
        if status not in _OPEN_ORDER_STATUSES:
            continue
        result.append(
            ReconciledOrder(
                order_id=str(row.get("order_id", "")),
                symbol=str(row.get("tradingsymbol", "")),
                status=status,
                qty=int(row.get("quantity", 0)),
                price=float(row.get("price", 0) or 0),
            )
        )
    return result


def _load_local(path: Path | None) -> list[LocalPositionSnapshot]:
    if path is None or not path.is_file():
        return []
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return [LocalPositionSnapshot.model_validate(item) for item in raw]


def _detect_drift(
    broker: list[Position],
    local: list[LocalPositionSnapshot],
    records: list[OrderRecord],
) -> list[str]:
    """Detect symbols whose broker view disagrees with our records.

    Drift sources, in priority order:

        * Legacy JSON ``local`` snapshot mismatch (qty/side differ).
        * ``ENTERED`` records with no broker position (closed externally).
        * Broker positions whose symbol is unknown to both local snapshot
          and state-store records (foreign positions).
    """
    drift: set[str] = set()

    if local:
        broker_map = {p.symbol: p for p in broker}
        local_map = {p.symbol: p for p in local}
        for symbol in sorted(local_map):
            loc = local_map[symbol]
            broker_pos = broker_map.get(symbol)
            if broker_pos is None or broker_pos.qty != loc.qty or broker_pos.side != loc.side:
                drift.add(symbol)
        for symbol in sorted(broker_map):
            if symbol not in local_map:
                drift.add(symbol)
        return sorted(drift)

    open_records_by_symbol = {
        r.symbol for r in records if r.state in OPEN_STATES
    }
    broker_symbols = {p.symbol for p in broker}
    for symbol in open_records_by_symbol - broker_symbols:
        drift.add(symbol)
    for symbol in broker_symbols - open_records_by_symbol:
        if not open_records_by_symbol:
            # No records at all → nothing to compare against; do not flag.
            continue
        drift.add(symbol)
    return sorted(drift)
