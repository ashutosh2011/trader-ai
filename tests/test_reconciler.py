"""Tests for the state-store-driven :class:`StateReconciler`."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from execution.broker import Position
from execution.order_state import OrderRecord, OrderState, OrderStateStore
from execution.reconciler import LocalPositionSnapshot, StateReconciler

IST = ZoneInfo("Asia/Kolkata")


def _bare_position(symbol: str = "RELIANCE", *, qty: int = 10) -> Position:
    return Position(
        symbol=symbol,
        side="LONG",
        qty=qty,
        entry_price=2500.0,
        stop_loss=None,
        target=None,
        strategy_id="kite_sync",
        opened_at=datetime(2024, 1, 1, 10, 0, tzinfo=IST),
    )


def _record(
    *,
    client_order_id: str = "tb-aaaa",
    symbol: str = "RELIANCE",
    stop_loss: float = 2400.0,
    target: float = 2600.0,
    state: OrderState = OrderState.ENTERED,
) -> OrderRecord:
    now = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    return OrderRecord(
        client_order_id=client_order_id,
        symbol=symbol,
        side="BUY",
        qty=10,
        entry_price=2500.0,
        stop_loss=stop_loss,
        target=target,
        state=state,
        entry_order_id="1000",
        sl_gtt_id=50_000,
        fill_price=2500.0,
        signal_ts=now,
        created_at=now,
        updated_at=now,
        strategy_id="ema_crossover",
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[OrderStateStore]:
    s = OrderStateStore(tmp_path / "state.duckdb")
    try:
        yield s
    finally:
        s.close()


def test_reconcile_no_records_leaves_levels_none(store: OrderStateStore) -> None:
    broker = MagicMock()
    broker.get_positions.return_value = [_bare_position()]
    reconciler = StateReconciler(broker, state_store=store)
    state = reconciler.reconcile(orders=[])
    assert len(state.positions) == 1
    assert state.positions[0].stop_loss is None
    assert state.positions[0].target is None


def test_reconcile_backfills_from_state_store(store: OrderStateStore) -> None:
    broker = MagicMock()
    broker.get_positions.return_value = [_bare_position("RELIANCE")]
    store.upsert(_record(symbol="RELIANCE", stop_loss=2400.0, target=2600.0))

    reconciler = StateReconciler(broker, state_store=store)
    state = reconciler.reconcile()
    assert state.positions[0].stop_loss == 2400.0
    assert state.positions[0].target == 2600.0
    assert state.positions[0].strategy_id == "ema_crossover"


def test_reconcile_partial_backfill_two_symbols(store: OrderStateStore) -> None:
    broker = MagicMock()
    broker.get_positions.return_value = [
        _bare_position("RELIANCE"),
        _bare_position("INFY"),
    ]
    store.upsert(_record(client_order_id="tb-a", symbol="RELIANCE"))

    reconciler = StateReconciler(broker, state_store=store)
    state = reconciler.reconcile()
    by_symbol = {p.symbol: p for p in state.positions}
    assert by_symbol["RELIANCE"].stop_loss == 2400.0
    assert by_symbol["RELIANCE"].target == 2600.0
    assert by_symbol["INFY"].stop_loss is None
    assert by_symbol["INFY"].target is None


def test_reconcile_drift_record_without_position(store: OrderStateStore) -> None:
    """ENTERED record with no live position → drift."""
    broker = MagicMock()
    broker.get_positions.return_value = []
    store.upsert(_record(symbol="RELIANCE"))

    reconciler = StateReconciler(broker, state_store=store)
    state = reconciler.reconcile()
    assert "RELIANCE" in state.drift_symbols


def test_reconcile_drift_foreign_position(store: OrderStateStore) -> None:
    """Broker has a position we never tracked → drift."""
    broker = MagicMock()
    broker.get_positions.return_value = [_bare_position("FOREIGN")]
    store.upsert(_record(symbol="RELIANCE"))

    reconciler = StateReconciler(broker, state_store=store)
    state = reconciler.reconcile()
    assert "FOREIGN" in state.drift_symbols
    assert "RELIANCE" in state.drift_symbols


def test_reconcile_local_snapshot_drift(tmp_path: Path) -> None:
    """Legacy JSON snapshot drift detection still works."""
    local_path = tmp_path / "state.json"
    local_path.write_text(
        '[{"symbol": "RELIANCE", "side": "LONG", "qty": 5, '
        '"entry_price": 2500.0, "strategy_id": "s1"}]',
        encoding="utf-8",
    )
    broker = MagicMock()
    broker.get_positions.return_value = [
        Position(
            symbol="RELIANCE",
            side="LONG",
            qty=10,
            entry_price=2500.0,
            stop_loss=2400.0,
            target=2600.0,
            strategy_id="s1",
            opened_at=datetime(2024, 1, 1, 10, 0, tzinfo=IST),
        )
    ]
    reconciler = StateReconciler(broker, local_state_path=local_path)
    state = reconciler.reconcile()
    assert "RELIANCE" in state.drift_symbols


def test_local_position_snapshot_basic() -> None:
    snap = LocalPositionSnapshot(
        symbol="X",
        side="LONG",
        qty=1,
        entry_price=100.0,
        strategy_id="s",
    )
    assert snap.symbol == "X"


def test_reconcile_ambiguous_multiple_records(store: OrderStateStore) -> None:
    """Two ENTERED records on same symbol — reconciler picks most recent."""
    broker = MagicMock()
    broker.get_positions.return_value = [_bare_position("RELIANCE")]
    older = _record(client_order_id="tb-older", symbol="RELIANCE")
    older.updated_at = datetime(2024, 1, 1, 9, 0, tzinfo=IST)
    older.target = 2580.0
    store.upsert(older)
    newer = _record(
        client_order_id="tb-newer",
        symbol="RELIANCE",
        target=2620.0,
    )
    newer.updated_at = datetime(2024, 1, 1, 11, 0, tzinfo=IST)
    store.upsert(newer)

    reconciler = StateReconciler(broker, state_store=store)
    state = reconciler.reconcile()
    assert state.positions[0].target == 2620.0


def test_reconcile_open_orders_diagnostics(store: OrderStateStore) -> None:
    broker = MagicMock()
    broker.get_positions.return_value = []
    reconciler = StateReconciler(broker, state_store=store)
    state = reconciler.reconcile(
        orders=[
            {
                "order_id": "X1",
                "tradingsymbol": "RELIANCE",
                "status": "OPEN",
                "quantity": 5,
                "price": 2500.0,
            },
            {
                "order_id": "X2",
                "tradingsymbol": "INFY",
                "status": "COMPLETE",
                "quantity": 1,
                "price": 1500.0,
            },
        ]
    )
    assert len(state.open_orders) == 1
    assert state.open_orders[0].order_id == "X1"
