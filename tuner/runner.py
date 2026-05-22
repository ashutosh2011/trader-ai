"""End-to-end tuning run orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import structlog

from tuner.active import ActiveConfigStore
from tuner.llm_tuner import LLMTuner, TunerMeta
from tuner.performance import StrategySymbolPerformance, collect_performance
from tuner.prompt import TuningContext
from tuner.schema import TuningPlan
from tuner.store import TuningStore

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TuningRunRecord:
    """Result of one tuning run."""

    run_id: str
    plan: TuningPlan
    meta: TunerMeta
    performances: list[StrategySymbolPerformance]


class TuningRunner:
    """Collect performance, call LLM, persist recommendations."""

    def __init__(
        self,
        llm_tuner: LLMTuner,
        tuning_store: TuningStore,
        active_store: ActiveConfigStore,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        self._tuner = llm_tuner
        self._store = tuning_store
        self._active = active_store
        self._conn = conn

    async def run(
        self,
        ctx: TuningContext,
        *,
        lookback_days: int = 30,
        max_runs: int = 50,
        run_id: str | None = None,
    ) -> TuningRunRecord:
        """Execute one tuning review."""
        active_lookup = self._active.as_lookup()
        performances = collect_performance(
            self._conn,
            lookback_days=lookback_days,
            max_runs=max_runs,
            active_configs=active_lookup,
        )
        plan, meta = await self._tuner.generate(performances, ctx)
        rid = self._store.record_run(
            plan=plan,
            meta=meta,
            pairs_reviewed=len(performances),
            run_id=run_id,
        )
        logger.info(
            "tuning_run_complete",
            run_id=rid,
            status=meta.status,
            recommendations=len(plan.recommendations),
        )
        return TuningRunRecord(
            run_id=rid,
            plan=plan,
            meta=meta,
            performances=performances,
        )
