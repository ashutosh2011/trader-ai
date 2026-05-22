"""Tests for TuningStore persistence."""

from __future__ import annotations

from pathlib import Path

import duckdb

from tuner.llm_tuner import TunerMeta
from tuner.schema import SymbolTuningRecommendation, TuningPlan
from tuner.store import TuningStore


def test_record_and_fetch(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "d.duckdb"))
    store = TuningStore(conn)
    plan = TuningPlan(
        name="Test",
        summary_rationale="because",
        recommendations=[
            SymbolTuningRecommendation(
                symbol="INFY",
                current_strategy_id="ema_crossover",
                action="keep",
                rationale="ok",
                confidence=0.5,
            )
        ],
    )
    meta = TunerMeta(status="ok", provider="stub", latency_ms=1)
    run_id = store.record_run(plan=plan, meta=meta, pairs_reviewed=1)
    detail = store.get_run(run_id)
    assert detail is not None
    assert detail.plan.name == "Test"
    assert len(detail.recommendations) == 1
    assert detail.recommendations[0].status == "pending"
    conn.close()
