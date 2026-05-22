"""DuckDB persistence for tuning runs and recommendations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import duckdb
import structlog

from tuner.llm_tuner import TunerMeta, TunerMetaStatus
from tuner.schema import (
    RecommendationStatus,
    StoredRecommendation,
    SymbolTuningRecommendation,
    TuningAction,
    TuningPlan,
)

logger = structlog.get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")

TUNING_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tuning_runs (
    id VARCHAR PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    name VARCHAR NOT NULL,
    summary_rationale VARCHAR NOT NULL,
    plan_json VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    provider VARCHAR,
    llm_latency_ms INTEGER,
    pairs_reviewed INTEGER NOT NULL,
    recommendation_count INTEGER NOT NULL,
    error VARCHAR
);
"""

TUNING_RECOMMENDATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tuning_recommendations (
    id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    current_strategy_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    recommended_strategy_id VARCHAR,
    params_json VARCHAR NOT NULL,
    rationale VARCHAR NOT NULL,
    confidence DOUBLE NOT NULL,
    status VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ
);
"""


@dataclass(frozen=True)
class TuningRunSummary:
    id: str
    created_at: datetime
    name: str
    summary_rationale: str
    status: TunerMetaStatus
    provider: str | None
    llm_latency_ms: int | None
    pairs_reviewed: int
    recommendation_count: int
    error: str | None


@dataclass(frozen=True)
class TuningRunDetail:
    summary: TuningRunSummary
    plan: TuningPlan
    recommendations: list[StoredRecommendation]


class TuningStore:
    """Persist tuning runs and per-symbol recommendations."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self._conn.execute(TUNING_RUNS_SCHEMA)
        self._conn.execute(TUNING_RECOMMENDATIONS_SCHEMA)

    def record_run(
        self,
        *,
        plan: TuningPlan,
        meta: TunerMeta,
        pairs_reviewed: int,
        run_id: str | None = None,
    ) -> str:
        rid = run_id or uuid4().hex[:12]
        created_at = datetime.now(tz=IST)
        plan_json = plan.model_dump_json()
        self._conn.execute(
            "INSERT INTO tuning_runs ("
            "id, created_at, name, summary_rationale, plan_json, status, "
            "provider, llm_latency_ms, pairs_reviewed, recommendation_count, error"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rid,
                created_at,
                plan.name,
                plan.summary_rationale,
                plan_json,
                meta.status,
                meta.provider if meta.status == "ok" else meta.provider,
                meta.latency_ms,
                pairs_reviewed,
                len(plan.recommendations),
                meta.error,
            ],
        )
        for rec in plan.recommendations:
            self._insert_recommendation(run_id=rid, rec=rec, created_at=created_at)
        logger.info(
            "tuning_run_persisted",
            run_id=rid,
            status=meta.status,
            recommendations=len(plan.recommendations),
        )
        return rid

    def _insert_recommendation(
        self,
        *,
        run_id: str,
        rec: SymbolTuningRecommendation,
        created_at: datetime,
    ) -> str:
        rec_id = uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO tuning_recommendations ("
            "id, run_id, symbol, current_strategy_id, action, "
            "recommended_strategy_id, params_json, rationale, confidence, "
            "status, created_at, resolved_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rec_id,
                run_id,
                rec.symbol,
                rec.current_strategy_id,
                rec.action,
                rec.recommended_strategy_id,
                json.dumps(rec.params),
                rec.rationale,
                rec.confidence,
                "pending",
                created_at,
                None,
            ],
        )
        return rec_id

    def list_runs(self, limit: int = 20) -> list[TuningRunSummary]:
        rows = self._conn.execute(
            "SELECT id, created_at, name, summary_rationale, status, provider, "
            "llm_latency_ms, pairs_reviewed, recommendation_count, error "
            "FROM tuning_runs ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        return [_row_summary(r) for r in rows]

    def get_run(self, run_id: str) -> TuningRunDetail | None:
        row = self._conn.execute(
            "SELECT id, created_at, name, summary_rationale, plan_json, status, "
            "provider, llm_latency_ms, pairs_reviewed, recommendation_count, error "
            "FROM tuning_runs WHERE id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None
        summary = _row_summary(row)
        plan = TuningPlan.model_validate_json(str(row[4]))
        recs = self.list_recommendations(run_id)
        return TuningRunDetail(summary=summary, plan=plan, recommendations=recs)

    def list_recommendations(self, run_id: str) -> list[StoredRecommendation]:
        rows = self._conn.execute(
            "SELECT id, run_id, symbol, current_strategy_id, action, "
            "recommended_strategy_id, params_json, rationale, confidence, "
            "status, created_at, resolved_at "
            "FROM tuning_recommendations WHERE run_id = ? ORDER BY symbol",
            [run_id],
        ).fetchall()
        return [_row_rec(r) for r in rows]

    def get_recommendation(self, rec_id: str) -> StoredRecommendation | None:
        row = self._conn.execute(
            "SELECT id, run_id, symbol, current_strategy_id, action, "
            "recommended_strategy_id, params_json, rationale, confidence, "
            "status, created_at, resolved_at "
            "FROM tuning_recommendations WHERE id = ?",
            [rec_id],
        ).fetchone()
        if row is None:
            return None
        return _row_rec(row)

    def set_recommendation_status(
        self,
        rec_id: str,
        status: RecommendationStatus,
    ) -> StoredRecommendation | None:
        resolved_at = datetime.now(tz=IST) if status != "pending" else None
        self._conn.execute(
            "UPDATE tuning_recommendations SET status = ?, resolved_at = ? "
            "WHERE id = ?",
            [status, resolved_at, rec_id],
        )
        return self.get_recommendation(rec_id)


def _row_summary(row: tuple[Any, ...]) -> TuningRunSummary:
    created_at = row[1]
    if not isinstance(created_at, datetime):
        created_at = datetime.fromisoformat(str(created_at))
    return TuningRunSummary(
        id=str(row[0]),
        created_at=created_at,
        name=str(row[2]),
        summary_rationale=str(row[3]),
        status=cast(TunerMetaStatus, row[5]),
        provider=str(row[6]) if row[6] is not None else None,
        llm_latency_ms=int(row[7]) if row[7] is not None else None,
        pairs_reviewed=int(row[8]),
        recommendation_count=int(row[9]),
        error=str(row[10]) if row[10] is not None else None,
    )


def _row_rec(row: tuple[Any, ...]) -> StoredRecommendation:
    created_at = row[10]
    resolved = row[11]
    if not isinstance(created_at, datetime):
        created_at = datetime.fromisoformat(str(created_at))
    resolved_str = (
        resolved.isoformat()
        if isinstance(resolved, datetime)
        else (str(resolved) if resolved is not None else None)
    )
    return StoredRecommendation(
        id=str(row[0]),
        run_id=str(row[1]),
        symbol=str(row[2]),
        current_strategy_id=str(row[3]),
        action=cast(TuningAction, row[4]),
        recommended_strategy_id=(
            str(row[5]) if row[5] is not None else None
        ),
        params=json.loads(str(row[6])),
        rationale=str(row[7]),
        confidence=float(row[8]),
        status=cast(RecommendationStatus, row[9]),
        created_at=created_at.isoformat(),
        resolved_at=resolved_str,
    )
