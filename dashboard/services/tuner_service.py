"""Dashboard / CLI bridge for the strategy tuner."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import structlog

from analyst.provider import LLMProvider
from analyst.providers.anthropic import AnthropicProvider
from analyst.providers.google import GoogleProvider
from analyst.providers.mock import MockLLMProvider
from analyst.providers.openai import OpenAIProvider
from config.settings import AppSettings
from dashboard.state import AppState
from tuner.active import ActiveConfigStore
from tuner.apply import apply_recommendation
from tuner.llm_tuner import LLMTuner
from tuner.prompt import TuningContext
from tuner.runner import TuningRunner, TuningRunRecord
from tuner.schema import StoredRecommendation, TuningAction
from tuner.store import TuningStore

logger = structlog.get_logger(__name__)

TunerProviderName = Literal["openai", "anthropic", "google", "stub"]

PROVIDER_OPTIONS: tuple[TunerProviderName, ...] = (
    "stub",
    "openai",
    "anthropic",
    "google",
)

_STUB_PLAN_JSON = json.dumps(
    {
        "name": "Stub tuning review",
        "summary_rationale": "Stub provider: deterministic sample adjustment.",
        "recommendations": [],
    }
)


def build_stub_provider() -> LLMProvider:
    return MockLLMProvider(_STUB_PLAN_JSON, name="stub")


def build_provider(name: TunerProviderName, settings: AppSettings) -> LLMProvider:
    if name == "stub":
        return build_stub_provider()
    if name == "anthropic":
        return AnthropicProvider(settings.analyst)
    if name == "openai":
        return OpenAIProvider(settings.analyst)
    if name == "google":
        return GoogleProvider(settings.analyst)
    msg = f"unknown tuner provider: {name}"
    raise ValueError(msg)


def action_label(action: TuningAction) -> str:
    labels = {
        "keep": "Keep",
        "modify_params": "Modify params",
        "switch_strategy": "Switch strategy",
        "disable": "Disable symbol",
    }
    return labels.get(action, action)


class TunerService:
    """Facade for tuning runs and applying recommendations."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def _runner(self, provider_name: TunerProviderName) -> TuningRunner:
        conn = self._state.dashboard_conn()
        store = TuningStore(conn)
        active = ActiveConfigStore(conn)
        provider = build_provider(provider_name, self._state.settings)
        return TuningRunner(
            LLMTuner(provider),
            store,
            active,
            conn,
        )

    def tuning_store(self) -> TuningStore:
        return TuningStore(self._state.dashboard_conn())

    def active_store(self) -> ActiveConfigStore:
        return ActiveConfigStore(self._state.dashboard_conn())

    async def run(
        self,
        *,
        provider_name: TunerProviderName,
        notes: str = "",
        lookback_days: int = 30,
    ) -> TuningRunRecord:
        ctx = TuningContext(
            as_of=datetime.now(tz=ZoneInfo("Asia/Kolkata")),
            notes=notes,
            lookback_days=lookback_days,
        )
        runner = self._runner(provider_name)
        return await runner.run(ctx, lookback_days=lookback_days)

    def apply_recommendation(self, rec_id: str) -> StoredRecommendation:
        store = self.tuning_store()
        active = self.active_store()
        stored = store.get_recommendation(rec_id)
        if stored is None:
            msg = f"recommendation not found: {rec_id}"
            raise ValueError(msg)
        if stored.status != "pending":
            msg = f"recommendation already {stored.status}"
            raise ValueError(msg)
        apply_recommendation(active, stored)
        updated = store.set_recommendation_status(rec_id, "applied")
        if updated is None:
            msg = f"failed to update recommendation: {rec_id}"
            raise ValueError(msg)
        return updated

    def reject_recommendation(self, rec_id: str) -> StoredRecommendation:
        store = self.tuning_store()
        stored = store.get_recommendation(rec_id)
        if stored is None:
            msg = f"recommendation not found: {rec_id}"
            raise ValueError(msg)
        if stored.status != "pending":
            msg = f"recommendation already {stored.status}"
            raise ValueError(msg)
        updated = store.set_recommendation_status(rec_id, "rejected")
        if updated is None:
            msg = f"failed to update recommendation: {rec_id}"
            raise ValueError(msg)
        return updated


def render_run_table(record: TuningRunRecord) -> str:
    lines = [
        f"Run {record.run_id} — {record.plan.name}",
        (
            f"status={record.meta.status} provider={record.meta.provider} "
            f"pairs={len(record.performances)} "
            f"recs={len(record.plan.recommendations)}"
        ),
        f"Rationale: {record.plan.summary_rationale}",
    ]
    if not record.plan.recommendations:
        lines.append("(no recommendations — all keep or fallback empty)")
    for rec in record.plan.recommendations:
        lines.append(
            f"  {rec.symbol}: {rec.action} -> "
            f"{rec.recommended_strategy_id or rec.current_strategy_id} "
            f"params={rec.params} ({rec.confidence:.0%})"
        )
    return "\n".join(lines)


def render_run_json(record: TuningRunRecord) -> str:
    return json.dumps(
        {
            "run_id": record.run_id,
            "plan": json.loads(record.plan.model_dump_json()),
            "meta": record.meta.model_dump(),
            "pairs_reviewed": len(record.performances),
        },
        indent=2,
    )
