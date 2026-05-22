"""End-to-end tuning runner tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pytest

from analyst.providers.mock import MockLLMProvider
from dashboard.state import BACKTEST_RUNS_SCHEMA
from tuner.active import ActiveConfigStore
from tuner.llm_tuner import LLMTuner
from tuner.prompt import TuningContext
from tuner.runner import TuningRunner
from tuner.store import TuningStore

IST = ZoneInfo("Asia/Kolkata")


@pytest.mark.asyncio
async def test_runner_persists_run(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "d.duckdb"))
    conn.execute(BACKTEST_RUNS_SCHEMA)
    run_at = datetime.now(tz=IST)
    conn.execute(
        "INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "bt1",
            "ema_crossover",
            "SYNTH",
            json.dumps({"strategy": {}, "source": {}}),
            50,
            run_at,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            json.dumps({"closed_trades": [], "equity_curve": [], "metrics": {}}),
        ],
    )
    plan_json = json.dumps(
        {
            "name": "Stub",
            "summary_rationale": "x",
            "recommendations": [],
        }
    )
    provider = MockLLMProvider(plan_json, name="stub")
    runner = TuningRunner(
        LLMTuner(provider),
        TuningStore(conn),
        ActiveConfigStore(conn),
        conn,
    )
    record = await runner.run(TuningContext(as_of=run_at, notes=""))
    assert record.run_id
    detail = TuningStore(conn).get_run(record.run_id)
    assert detail is not None
    conn.close()
