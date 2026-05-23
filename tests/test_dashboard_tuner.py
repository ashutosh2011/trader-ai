"""Dashboard HTTP tests for the strategy tuner."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.state import BACKTEST_RUNS_SCHEMA, AppState

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.5\n", encoding="utf-8")
    state = AppState(
        settings=AppSettings.model_validate({}),
        config_path=config_path,
        env_path=tmp_path / ".env",
        dashboard_db_path=tmp_path / "dash.duckdb",
        journal_path=None,
    )
    conn = state.dashboard_conn()
    conn.execute(BACKTEST_RUNS_SCHEMA)
    run_at = datetime.now(tz=IST)
    # NOTE: backtest_runs grew a ``group_id`` column in the multi-strategy
    # redesign; we now name the legacy columns explicitly so this seed row
    # stays compatible with both the pre- and post-redesign schema.
    conn.execute(
        "INSERT INTO backtest_runs ("
        "id, strategy, symbol, params, bars_count, run_at, total_pnl, sharpe, "
        "win_rate, mdd_pct, total_trades, result_json"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "bt1",
            "ema_crossover",
            "INFY",
            json.dumps({"strategy": {"fast_period": 12}, "source": {}}),
            100,
            run_at,
            5.0,
            0.1,
            60.0,
            2.0,
            2,
            json.dumps(
                {
                    "closed_trades": [
                        {
                            "symbol": "INFY",
                            "side": "LONG",
                            "pnl": 5.0,
                            "entry_price": 100.0,
                            "exit_price": 105.0,
                            "exit_reason": "target",
                        }
                    ],
                    "equity_curve": [],
                    "metrics": {},
                }
            ),
        ],
    )
    app = create_app(state=state)
    with TestClient(app) as test_client:
        yield test_client
    state.close()


def test_tuner_page_200(client: TestClient) -> None:
    resp = client.get("/tuner")
    assert resp.status_code == 200
    assert "run tuning review" in resp.text


def test_tuner_run_stub(client: TestClient) -> None:
    resp = client.post(
        "/api/tuner/run",
        json={"provider": "stub", "lookback_days": 30, "notes": ""},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    detail = client.get(f"/tuner/{run_id}")
    assert detail.status_code == 200
