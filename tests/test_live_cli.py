from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

import orchestrator.main as orch_main
from execution.order_state import OrderStateStore
from orchestrator.main import cli
from tests.fixtures.bars import make_synthetic_bars


def test_cli_live_dry_run_default_works() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["live", "--bars-count", "100", "--dry-run"])
    assert result.exit_code == 0
    assert "live signals=" in result.output


def test_cli_live_kill_switch_env() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["live", "--bars-count", "50"],
        env={"KILL_SWITCH": "1"},
    )
    assert result.exit_code == 1
    assert "KILL switch" in result.output


def test_cli_live_no_dry_run_synthetic_refused() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["live", "--bars-count", "50", "--no-dry-run"])
    assert result.exit_code != 0
    assert "requires a real bar source" in result.output


def test_cli_live_no_dry_run_with_bars_requires_allow_replay_live(tmp_path: Path) -> None:
    bars_path = tmp_path / "bars.csv"
    frame: pd.DataFrame = make_synthetic_bars(50, seed=1)
    frame.to_csv(bars_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["live", "--bars", str(bars_path), "--no-dry-run"],
    )
    assert result.exit_code != 0
    assert "--allow-replay-live" in result.output


def test_cli_live_no_dry_run_with_allow_replay_live_proceeds(tmp_path: Path) -> None:
    bars_path = tmp_path / "bars.csv"
    frame: pd.DataFrame = make_synthetic_bars(50, seed=1)
    frame.to_csv(bars_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "live",
            "--bars", str(bars_path),
            "--no-dry-run",
            "--allow-replay-live",
        ],
    )
    assert result.exit_code == 0
    assert "WARNING" in result.output
    assert "live signals=" in result.output


def test_cli_live_kite_feed_without_credentials_fails() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["live", "--live-feed", "kite", "--no-dry-run"],
        env={"KITE_API_KEY": "", "KITE_ACCESS_TOKEN": ""},
    )
    assert result.exit_code != 0
    assert "KITE_API_KEY" in result.output


def test_cli_live_qty_zero_rejected() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["live", "--bars-count", "50", "--qty", "0", "--dry-run"])
    assert result.exit_code != 0
    assert "--qty 0" in result.output


def test_cli_live_state_db_flag_accepted(tmp_path: Path) -> None:
    """The --state-db option is wired through and parses (dry-run path)."""
    db = tmp_path / "state.duckdb"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "live",
            "--bars-count", "50",
            "--dry-run",
            "--state-db", str(db),
        ],
    )
    assert result.exit_code == 0
    assert "live signals=" in result.output


def test_cli_flatten_requires_kite_credentials() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["flatten"],
        env={"KITE_API_KEY": "", "KITE_ACCESS_TOKEN": ""},
    )
    assert result.exit_code != 0
    assert "flatten requires" in result.output


def test_cli_flatten_invokes_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end wiring of the ``flatten`` subcommand with a mocked broker."""
    db = tmp_path / "state.duckdb"
    flatten_calls: list[int] = []

    class _StubKiteClient:
        @classmethod
        def from_settings(cls, _settings: Any) -> "_StubKiteClient":
            return cls()

    class _StubKiteBroker:
        def __init__(self, _client: Any, **_: Any) -> None:
            pass

        def flatten_all(self) -> None:
            flatten_calls.append(1)

    monkeypatch.setattr(orch_main, "KiteClient", _StubKiteClient)
    monkeypatch.setattr(orch_main, "KiteBroker", _StubKiteBroker)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["flatten", "--state-db", str(db)],
        env={"KITE_API_KEY": "fake_key", "KITE_ACCESS_TOKEN": "fake_token"},
    )
    assert result.exit_code == 0, result.output
    assert flatten_calls == [1]
    assert "flatten complete" in result.output
    # State DB file exists post-run (created by OrderStateStore even if empty).
    assert db.is_file()


def test_cli_flatten_reports_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "state.duckdb"

    from datetime import datetime
    from zoneinfo import ZoneInfo

    from execution.broker import FlattenIncomplete, Position

    ist = ZoneInfo("Asia/Kolkata")
    stuck = Position(
        symbol="RELIANCE",
        side="LONG",
        qty=1,
        entry_price=2500.0,
        strategy_id="kite_sync",
        opened_at=datetime(2024, 1, 1, 10, 0, tzinfo=ist),
    )

    class _StubKiteClient:
        @classmethod
        def from_settings(cls, _settings: Any) -> "_StubKiteClient":
            return cls()

    class _StubKiteBroker:
        def __init__(self, _client: Any, **_: Any) -> None:
            pass

        def flatten_all(self) -> None:
            raise FlattenIncomplete(open_positions=[stuck], attempts=10)

    monkeypatch.setattr(orch_main, "KiteClient", _StubKiteClient)
    monkeypatch.setattr(orch_main, "KiteBroker", _StubKiteBroker)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["flatten", "--state-db", str(db)],
        env={"KITE_API_KEY": "fake_key", "KITE_ACCESS_TOKEN": "fake_token"},
    )
    assert result.exit_code == 2
    assert "RELIANCE" in result.output


def test_state_summary_prints_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "state.duckdb"
    store = OrderStateStore(db)
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from execution.order_state import OrderRecord, OrderState

        ist = ZoneInfo("Asia/Kolkata")
        now = datetime(2024, 1, 1, 10, 0, tzinfo=ist)
        store.upsert(
            OrderRecord(
                client_order_id="tb-a",
                symbol="RELIANCE",
                side="BUY",
                qty=10,
                entry_price=2500.0,
                stop_loss=2480.0,
                target=2540.0,
                state=OrderState.PENDING_ENTRY,
                signal_ts=now,
                created_at=now,
                updated_at=now,
                strategy_id="ema",
            )
        )
        orch_main._print_state_summary(store, db)
        captured = capsys.readouterr()
        assert "records=1" in captured.out
        assert "PENDING_ENTRY=1" in captured.out
    finally:
        store.close()
