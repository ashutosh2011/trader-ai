from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from execution.broker import Position
from execution.reconciler import LocalPositionSnapshot, StateReconciler

IST = ZoneInfo("Asia/Kolkata")


def test_reconcile_no_drift() -> None:
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
    reconciler = StateReconciler(broker)
    state = reconciler.reconcile(orders=[])
    assert len(state.positions) == 1
    assert state.drift_symbols == []


def test_reconcile_detects_drift(tmp_path: Path) -> None:
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


def test_local_position_snapshot() -> None:
    snap = LocalPositionSnapshot(
        symbol="X",
        side="LONG",
        qty=1,
        entry_price=100.0,
        strategy_id="s",
    )
    assert snap.symbol == "X"
