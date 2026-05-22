"""Unit tests for dashboard services (no FastAPI required)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pytest

from config.settings import AppSettings
from dashboard.services.backtest_runner import BacktestRunner
from dashboard.services.config_io import read_config_text, save_yaml, validate_yaml
from dashboard.services.journal_reader import JournalReader
from dashboard.services.kill_switch import KillSwitchService
from dashboard.services.orders import OrdersService
from dashboard.services.strategy_state import StrategyStateService
from dashboard.state import BACKTEST_RUNS_SCHEMA, STRATEGY_SETTINGS_SCHEMA
from data.kite_client import KiteClient
from execution.order_state import OrderRecord, OrderState, OrderStateStore

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_lifecycle(tmp_path: Path) -> None:
    kill_file = tmp_path / "KILL"
    service = KillSwitchService(kill_file, env_var="KILL_SWITCH_NOPE")
    assert service.is_active() is False

    service.enable()
    assert kill_file.is_file()
    assert service.is_active() is True

    service.disable()
    assert kill_file.is_file() is False
    assert service.is_active() is False


def test_kill_switch_set_idempotent(tmp_path: Path) -> None:
    kill_file = tmp_path / "KILL"
    service = KillSwitchService(kill_file, env_var="KILL_SWITCH_NOPE")
    service.set(True)
    service.set(True)
    assert service.is_active() is True
    service.set(False)
    service.set(False)
    assert service.is_active() is False


# ---------------------------------------------------------------------------
# config io
# ---------------------------------------------------------------------------


_VALID_YAML = """\
risk:
  max_loss_per_trade_pct: 0.7
  daily_loss_cap_pct: 2.5
  max_open_positions: 5
"""

_INVALID_YAML = """\
risk:
  max_loss_per_trade_pct: -1.0
"""

_BAD_SYNTAX = "risk:\n  max_loss: : :\n"


def test_validate_yaml_ok() -> None:
    result = validate_yaml(_VALID_YAML)
    assert result.ok is True
    assert result.issues == []
    assert result.parsed is not None
    assert result.parsed["risk"]["max_loss_per_trade_pct"] == 0.7


def test_validate_yaml_invalid_structure() -> None:
    result = validate_yaml(_INVALID_YAML)
    assert result.ok is False
    assert result.issues
    assert any("max_loss" in issue.location for issue in result.issues)


def test_validate_yaml_bad_syntax() -> None:
    result = validate_yaml(_BAD_SYNTAX)
    assert result.ok is False
    assert result.issues
    assert result.issues[0].type == "yaml_error"


def test_save_yaml_writes_and_backs_up(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.1\n")

    result = save_yaml(_VALID_YAML, config_path=config_path)
    assert result.ok is True
    assert result.backup_path is not None and result.backup_path.is_file()
    assert "max_loss_per_trade_pct: 0.1" in result.backup_path.read_text(encoding="utf-8")
    assert "max_loss_per_trade_pct: 0.7" in config_path.read_text(encoding="utf-8")


def test_save_yaml_invalid_does_not_write(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.1\n")

    result = save_yaml(_INVALID_YAML, config_path=config_path)
    assert result.ok is False
    assert result.backup_path is None
    assert "max_loss_per_trade_pct: 0.1" in config_path.read_text(encoding="utf-8")


def test_read_config_text_falls_back_to_example(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"
    example = tmp_path / "example.yaml"
    example.write_text("hello: world\n")
    assert read_config_text(config_path, example) == "hello: world\n"


# ---------------------------------------------------------------------------
# journal reader
# ---------------------------------------------------------------------------


def _write_journal(path: Path, entries: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_journal_reader_tail_filters_event_type(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    _write_journal(
        path,
        [
            {"ts": "2026-05-21T10:00:00+05:30", "event": "signal", "symbol": "FOO"},
            {"ts": "2026-05-21T10:00:01+05:30", "event": "order", "order": {"symbol": "FOO"}},
            {"ts": "2026-05-21T10:00:02+05:30", "event": "signal", "symbol": "BAR"},
        ],
    )
    reader = JournalReader(path)
    result = reader.tail(limit=10, event_types=["signal"])
    assert [e.event for e in result.entries] == ["signal", "signal"]
    assert {e.symbol for e in result.entries} == {"FOO", "BAR"}


def test_journal_reader_since_ts(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    _write_journal(
        path,
        [
            {"ts": "2026-05-21T10:00:00+05:30", "event": "signal"},
            {"ts": "2026-05-21T10:00:05+05:30", "event": "signal"},
        ],
    )
    reader = JournalReader(path)
    result = reader.tail(limit=10, since_ts="2026-05-21T10:00:00+05:30")
    assert len(result.entries) == 1
    assert result.entries[0].ts == "2026-05-21T10:00:05+05:30"


def test_journal_reader_symbol_filter(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    _write_journal(
        path,
        [
            {"ts": "2026-05-21T10:00:00+05:30", "event": "order", "order": {"symbol": "FOO"}},
            {"ts": "2026-05-21T10:00:01+05:30", "event": "order", "order": {"symbol": "BAR"}},
        ],
    )
    reader = JournalReader(path)
    result = reader.tail(limit=10, symbol="BAR")
    assert len(result.entries) == 1
    assert result.entries[0].symbol == "BAR"


def test_journal_reader_missing_file(tmp_path: Path) -> None:
    reader = JournalReader(tmp_path / "missing.jsonl")
    result = reader.tail(limit=10)
    assert result.entries == []


def test_journal_reader_today_realized_pnl(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    today = datetime.now().astimezone().date().isoformat()
    _write_journal(
        path,
        [
            {"ts": f"{today}T10:00:00+05:30", "event": "order", "pnl": 12.5},
            {"ts": f"{today}T10:01:00+05:30", "event": "order", "pnl": -3.0},
            {"ts": "2020-01-01T10:00:00+05:30", "event": "order", "pnl": 99.0},
        ],
    )
    reader = JournalReader(path)
    assert reader.today_realized_pnl() == pytest.approx(9.5)


# ---------------------------------------------------------------------------
# strategy state
# ---------------------------------------------------------------------------


def _dashboard_conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "dash.duckdb"))
    conn.execute(BACKTEST_RUNS_SCHEMA)
    conn.execute(STRATEGY_SETTINGS_SCHEMA)
    return conn


def test_strategy_state_default(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        svc = StrategyStateService(conn)
        assert svc.get("ema_crossover") is True
        assert svc.get("ema_crossover", default_enabled=False) is False
    finally:
        conn.close()


def test_strategy_state_set_get_toggle(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        svc = StrategyStateService(conn)
        svc.set("ema_crossover", enabled=False)
        assert svc.get("ema_crossover") is False
        setting = svc.toggle("ema_crossover")
        assert setting.enabled is True
        all_settings = svc.list_all()
        assert "ema_crossover" in all_settings
        assert all_settings["ema_crossover"].enabled is True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# backtest runner
# ---------------------------------------------------------------------------


def test_backtest_runner_run_and_list(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        run_id = runner.run(
            strategy_id="ema_crossover",
            symbol="SYNTH",
            bars_count=200,
            params={"fast_period": 5, "slow_period": 12, "atr_period": 7},
            qty=1,
        )
        assert isinstance(run_id, str) and len(run_id) > 0
        runs = runner.list_runs(limit=10)
        assert len(runs) == 1
        assert runs[0].id == run_id
        assert runs[0].strategy == "ema_crossover"
        detail = runner.get_run(run_id)
        assert detail is not None
        assert detail.summary.id == run_id
        assert isinstance(detail.equity_curve, list)
        assert "sharpe_ratio" in detail.metrics
    finally:
        conn.close()


def test_backtest_runner_run_with_kite_historical_data(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)

    class FakeKiteClient:
        def historical_data(
            self,
            instrument_token: int,
            from_date: datetime,
            to_date: datetime,
            interval: str,
        ) -> list[dict[str, object]]:
            assert instrument_token == 12345
            assert interval == "minute"
            timestamps = pd.date_range(
                start=from_date,
                periods=200,
                freq="1min",
                tz=IST,
            )
            rows: list[dict[str, object]] = []
            for idx, ts in enumerate(timestamps):
                close = 100.0 + (idx * 0.05)
                rows.append(
                    {
                        "date": ts.to_pydatetime(),
                        "open": close - 0.1,
                        "high": close + 0.2,
                        "low": close - 0.2,
                        "close": close,
                        "volume": 1000 + idx,
                    }
                )
            return rows

    settings = AppSettings.model_validate(
        {
            "kite": {
                "api_key": "key",
                "access_token": "token",
            },
            "data": {
                "duckdb_path": tmp_path / "candles.duckdb",
            },
        }
    )

    def fake_kite_factory(_settings: AppSettings) -> KiteClient:
        return cast(KiteClient, FakeKiteClient())

    try:
        runner = BacktestRunner(
            conn,
            settings=settings,
            kite_client_factory=fake_kite_factory,
        )
        run_id = runner.run(
            strategy_id="ema_crossover",
            symbol="RELIANCE",
            bars_count=50,
            params={"fast_period": 5, "slow_period": 12, "atr_period": 7},
            qty=1,
            data_source="kite",
            instrument_token=12345,
            timeframe="minute",
            from_date=datetime(2024, 1, 1, 9, 15, tzinfo=IST),
            to_date=datetime(2024, 1, 1, 12, 35, tzinfo=IST),
        )
        detail = runner.get_run(run_id)
        assert detail is not None
        assert detail.summary.symbol == "RELIANCE"
        assert detail.summary.bars_count == 200
        assert detail.summary.params["source"]["type"] == "kite"
        assert detail.summary.params["source"]["instrument_token"] == 12345
    finally:
        conn.close()


def test_backtest_runner_unknown_strategy(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        with pytest.raises(KeyError):
            runner.run(strategy_id="nope", symbol="X", bars_count=50)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# orders service
# ---------------------------------------------------------------------------


def _record(client_order_id: str, *, state: OrderState, symbol: str = "FOO") -> OrderRecord:
    now = datetime.now().astimezone()
    return OrderRecord(
        client_order_id=client_order_id,
        symbol=symbol,
        side="BUY",
        qty=1,
        entry_price=100.0,
        stop_loss=95.0,
        target=110.0,
        state=state,
        signal_ts=now,
        created_at=now,
        updated_at=now,
        strategy_id="ema_crossover",
    )


def _order_store(tmp_path: Path) -> OrderStateStore:
    store = OrderStateStore(tmp_path / "orders.duckdb")
    store.upsert(_record("a", state=OrderState.ENTERED, symbol="FOO"))
    store.upsert(_record("b", state=OrderState.PENDING_ENTRY, symbol="BAR"))
    store.upsert(_record("c", state=OrderState.EXITED, symbol="FOO"))
    return store


def test_orders_service_page_filters(tmp_path: Path) -> None:
    store = _order_store(tmp_path)
    try:
        service = OrdersService(store)
        page = service.page(state="ENTERED")
        assert len(page.rows) == 1
        assert page.rows[0].client_order_id == "a"

        page = service.page(symbol="foo")  # case-insensitive
        assert {r.symbol for r in page.rows} == {"FOO"}

        page = service.page(state="ALL", per_page=2)
        assert page.total == 3
        assert page.total_pages == 2
    finally:
        store.close()


def test_orders_service_mark_rejects_non_terminal(tmp_path: Path) -> None:
    store = _order_store(tmp_path)
    try:
        service = OrdersService(store)
        with pytest.raises(ValueError):
            service.mark("a", state=OrderState.ENTERED)
    finally:
        store.close()


def test_orders_service_mark_rejects_already_terminal(tmp_path: Path) -> None:
    store = _order_store(tmp_path)
    try:
        service = OrdersService(store)
        with pytest.raises(RuntimeError):
            service.mark("c", state=OrderState.FAILED)
    finally:
        store.close()


def test_orders_service_mark_open_record(tmp_path: Path) -> None:
    store = _order_store(tmp_path)
    try:
        service = OrdersService(store)
        updated = service.mark("a", state=OrderState.CANCELLED, reason="test_mark")
        assert updated is not None
        assert updated.state == OrderState.CANCELLED
        assert updated.error == "test_mark"
    finally:
        store.close()
