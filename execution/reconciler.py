"""Restart-time broker vs local state reconciliation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import structlog
from pydantic import BaseModel, ConfigDict, Field

from execution.broker import Broker, Position

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

PositionSide = Literal["LONG", "SHORT"]


class LocalPositionSnapshot(BaseModel):
    """Persisted local position row."""

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
    """Fetch broker state on startup and compare with local snapshots."""

    def __init__(self, broker: Broker, local_state_path: Path | None = None) -> None:
        self._broker = broker
        self._local_path = local_state_path

    def reconcile(self, *, orders: list[dict[str, Any]] | None = None) -> ReconciledState:
        """Reconcile broker positions/orders with optional local JSON state."""
        broker_positions = self._broker.get_positions()
        open_orders = _map_open_orders(orders or [])
        local = _load_local(self._local_path)
        drift = _detect_drift(broker_positions, local)
        notes: list[str] = []
        if drift:
            notes.append(f"position_drift: {','.join(drift)}")
            logger.warning("reconcile_drift", symbols=drift)
        state = ReconciledState(
            reconciled_at=datetime.now(tz=IST),
            positions=broker_positions,
            open_orders=open_orders,
            drift_symbols=drift,
            notes=notes,
        )
        logger.info(
            "reconcile_complete",
            positions=len(broker_positions),
            open_orders=len(open_orders),
            drift=len(drift),
        )
        return state


def _map_open_orders(rows: list[dict[str, Any]]) -> list[ReconciledOrder]:
    open_statuses = {"OPEN", "TRIGGER PENDING", "PUT ORDER REQ RECEIVED"}
    result: list[ReconciledOrder] = []
    for row in rows:
        status = str(row.get("status", ""))
        if status not in open_statuses:
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
) -> list[str]:
    if not local:
        return []
    broker_map = {p.symbol: p for p in broker}
    local_map = {p.symbol: p for p in local}
    drift: list[str] = []
    for symbol in sorted(local_map):
        loc = local_map[symbol]
        broker_pos = broker_map.get(symbol)
        if broker_pos is None:
            drift.append(symbol)
            continue
        if broker_pos.qty != loc.qty or broker_pos.side != loc.side:
            drift.append(symbol)
    for symbol in sorted(broker_map):
        if symbol not in local_map:
            drift.append(symbol)
    return sorted(set(drift))
