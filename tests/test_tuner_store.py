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


def test_legacy_null_recommendation_count_reads_as_zero(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "legacy.duckdb"))
    try:
        conn.execute(
            "CREATE TABLE tuning_runs ("
            "id VARCHAR PRIMARY KEY,"
            "created_at TIMESTAMPTZ NOT NULL,"
            "name VARCHAR NOT NULL,"
            "summary_rationale VARCHAR NOT NULL,"
            "plan_json VARCHAR NOT NULL,"
            "status VARCHAR NOT NULL,"
            "provider VARCHAR,"
            "llm_latency_ms INTEGER,"
            "pairs_reviewed INTEGER NOT NULL,"
            "recommendation_count INTEGER,"
            "error VARCHAR"
            ")"
        )
        store = TuningStore(conn)
        conn.execute(
            "INSERT INTO tuning_runs ("
            "id, created_at, name, summary_rationale, plan_json, status, "
            "provider, llm_latency_ms, pairs_reviewed, recommendation_count, error"
            ") VALUES (?, now(), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                "legacy-null",
                "Legacy",
                "old row",
                TuningPlan(
                    name="Legacy",
                    summary_rationale="old row",
                    recommendations=[],
                ).model_dump_json(),
                "ok",
                "stub",
                1,
                0,
                None,
                None,
            ],
        )

        # Re-creating the store simulates a dashboard restart and backfills
        # existing rows from older local DBs.
        store = TuningStore(conn)

        runs = store.list_runs()
        assert runs[0].id == "legacy-null"
        assert runs[0].recommendation_count == 0
        detail = store.get_run("legacy-null")
        assert detail is not None
        assert detail.summary.recommendation_count == 0
    finally:
        conn.close()
