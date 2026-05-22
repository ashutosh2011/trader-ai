"""Tests for trade performance aggregation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from dashboard.state import BACKTEST_RUNS_SCHEMA
from tuner.performance import collect_performance

IST = ZoneInfo("Asia/Kolkata")


def _insert_backtest(
    conn: duckdb.DuckDBPyConnection,
    *,
    strategy: str,
    symbol: str,
    trades: list[dict[str, object]],
) -> None:
    run_at = datetime.now(tz=IST)
    params = json.dumps({"strategy": {"rsi_period": 14}, "source": {}})
    result = json.dumps(
        {
            "closed_trades": trades,
            "equity_curve": [],
            "metrics": {},
        }
    )
    conn.execute(
        "INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "run1",
            strategy,
            symbol,
            params,
            100,
            run_at,
            sum(float(str(t["pnl"])) for t in trades),
            0.0,
            50.0,
            1.0,
            len(trades),
            result,
        ],
    )


def test_collect_groups_trades(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "d.duckdb"))
    conn.execute(BACKTEST_RUNS_SCHEMA)
    _insert_backtest(
        conn,
        strategy="rsi_mean_revert",
        symbol="INFY",
        trades=[
            {
                "symbol": "INFY",
                "side": "LONG",
                "pnl": -10.0,
                "entry_price": 100.0,
                "exit_price": 99.0,
                "exit_reason": "stop_loss",
            },
            {
                "symbol": "INFY",
                "side": "LONG",
                "pnl": 20.0,
                "entry_price": 100.0,
                "exit_price": 102.0,
                "exit_reason": "target",
            },
        ],
    )
    perfs = collect_performance(conn, lookback_days=30)
    assert len(perfs) == 1
    assert perfs[0].trade_count == 2
    assert perfs[0].total_pnl == 10.0
    assert perfs[0].win_count == 1
    conn.close()
