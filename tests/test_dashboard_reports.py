"""Tests for the reports service + /reports route + overview hero stats."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.services.reports import ReportsService
from dashboard.state import BACKTEST_RUNS_SCHEMA, AppState


def _seed_run(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    strategy: str,
    symbol: str,
    pnl: float,
    trades: int,
    run_at: datetime,
    win_rate: float = 50.0,
) -> None:
    conn.execute(
        "INSERT INTO backtest_runs ("
        "id, strategy, symbol, params, bars_count, run_at, total_pnl, sharpe, "
        "win_rate, mdd_pct, total_trades, result_json"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run_id,
            strategy,
            symbol,
            json.dumps({}),
            100,
            run_at,
            pnl,
            1.0,
            win_rate,
            2.0,
            trades,
            json.dumps({}),
        ],
    )


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    db = tmp_path / "dash.duckdb"
    c = duckdb.connect(str(db))
    c.execute(BACKTEST_RUNS_SCHEMA)
    try:
        yield c
    finally:
        c.close()


def test_by_strategy_sorts_by_pnl_desc(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime.now().astimezone()
    _seed_run(conn, run_id="a", strategy="ema", symbol="X", pnl=10.0, trades=2, run_at=now)
    _seed_run(conn, run_id="b", strategy="ema", symbol="Y", pnl=5.0, trades=1, run_at=now)
    _seed_run(conn, run_id="c", strategy="rsi", symbol="Z", pnl=20.0, trades=4, run_at=now)
    service = ReportsService(conn)
    stats = service.by_strategy()
    assert [s.strategy for s in stats] == ["rsi", "ema"]
    by_name = {s.strategy: s for s in stats}
    assert by_name["ema"].runs == 2
    assert by_name["ema"].total_pnl == 15.0
    assert by_name["ema"].total_trades == 3


def test_by_symbol(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime.now().astimezone()
    _seed_run(conn, run_id="a", strategy="s", symbol="X", pnl=10.0, trades=1, run_at=now)
    _seed_run(conn, run_id="b", strategy="s", symbol="X", pnl=5.0, trades=1, run_at=now)
    _seed_run(conn, run_id="c", strategy="s", symbol="Y", pnl=20.0, trades=2, run_at=now)
    service = ReportsService(conn)
    stats = service.by_symbol()
    by_sym = {s.symbol: s for s in stats}
    assert by_sym["X"].total_pnl == 15.0
    assert by_sym["X"].runs == 2
    assert by_sym["Y"].total_pnl == 20.0


def test_top_winners_and_losers(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime.now().astimezone()
    for i in range(6):
        _seed_run(
            conn,
            run_id=f"id{i}",
            strategy="s",
            symbol=f"S{i}",
            pnl=float(i - 2) * 10,
            trades=1,
            run_at=now,
        )
    service = ReportsService(conn)
    winners = service.top_winners(limit=3)
    assert [w.total_pnl for w in winners] == [30.0, 20.0, 10.0]
    losers = service.top_losers(limit=3)
    assert [r.total_pnl for r in losers] == [-20.0, -10.0, 0.0]


def test_overview_stats_windows(conn: duckdb.DuckDBPyConnection) -> None:
    now = datetime.now().astimezone()
    _seed_run(conn, run_id="r1", strategy="s", symbol="X", pnl=5.0, trades=1, run_at=now)
    _seed_run(
        conn, run_id="r2", strategy="s", symbol="X", pnl=7.0, trades=2,
        run_at=now - timedelta(days=2),
    )
    _seed_run(
        conn, run_id="r3", strategy="s", symbol="X", pnl=11.0, trades=3,
        run_at=now - timedelta(days=10),
    )
    _seed_run(
        conn, run_id="r4", strategy="s", symbol="X", pnl=13.0, trades=4,
        run_at=now - timedelta(days=60),
    )
    service = ReportsService(conn)
    stats = service.overview_stats(window_days=30)
    assert stats.total_backtests == 4
    assert stats.total_trades == 10
    assert stats.pnl_all_time == 36.0
    assert stats.pnl_7d == 12.0
    assert stats.pnl_30d == 23.0
    assert len(stats.sparkline) == 30


def test_overview_stats_empty(conn: duckdb.DuckDBPyConnection) -> None:
    service = ReportsService(conn)
    stats = service.overview_stats()
    assert stats.total_backtests == 0
    assert stats.pnl_all_time == 0.0
    assert len(stats.sparkline) == 30
    assert all(p.total_pnl == 0.0 for p in stats.sparkline)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.5\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n", encoding="utf-8")
    settings = AppSettings.model_validate({}).model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "REP_TEST_KILL",
            "state_db_path": tmp_path / "orders.duckdb",
        }
    )
    state = AppState(
        settings=settings,
        config_path=config_path,
        env_path=env_path,
        dashboard_db_path=tmp_path / "dash.duckdb",
        journal_path=None,
    )
    try:
        yield state
    finally:
        state.close()


@pytest.fixture
def client(app_state: AppState) -> Iterator[TestClient]:
    app = create_app(app_state, dev=True)
    with TestClient(app) as c:
        yield c


def test_reports_page_empty(client: TestClient) -> None:
    response = client.get("/reports")
    assert response.status_code == 200
    body = response.text
    assert "Reports" in body
    assert "by strategy" in body


def test_reports_page_with_data(client: TestClient, app_state: AppState) -> None:
    conn = app_state.dashboard_conn()
    _seed_run(
        conn,
        run_id="rep1",
        strategy="ema_crossover",
        symbol="RELIANCE",
        pnl=123.45,
        trades=5,
        run_at=datetime.now().astimezone(),
    )
    response = client.get("/reports")
    assert response.status_code == 200
    body = response.text
    assert "ema_crossover" in body
    assert "RELIANCE" in body


def test_overview_page_hero_stats(client: TestClient, app_state: AppState) -> None:
    conn = app_state.dashboard_conn()
    _seed_run(
        conn,
        run_id="ov1",
        strategy="s",
        symbol="X",
        pnl=42.0,
        trades=2,
        run_at=datetime.now().astimezone(),
    )
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "today P&amp;L" in body  # & is HTML-escaped in rendered output
    assert "backtests run" in body
    # Hero P&L numbers render via the inr_amount filter — verify currency glyph.
    assert "₹" in body
