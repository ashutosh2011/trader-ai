"""LLM strategy tuner with fallback ladder."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from analyst._parsing import extract_json_object
from analyst.provider import LLMProvider
from tuner.performance import StrategySymbolPerformance
from tuner.prompt import TuningContext, build_tuning_prompt
from tuner.schema import TuningPlan
from tuner.validate import sanitise_plan

logger = structlog.get_logger(__name__)

TUNER_TIMEOUT_SEC = 10.0
RAW_PREVIEW_CHARS = 300

TunerMetaStatus = Literal[
    "ok",
    "fallback_transport",
    "fallback_parse_error",
    "fallback_unexpected",
]


class TunerMeta(BaseModel):
    """Bookkeeping for one tuner LLM call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: TunerMetaStatus
    provider: str
    latency_ms: int = Field(ge=0)
    error: str | None = None
    raw_preview: str | None = None


DEFAULT_TUNING_PLAN = TuningPlan(
    name="Default fallback: no changes",
    summary_rationale=(
        "LLM unavailable; keeping all strategies unchanged. "
        "Re-run when the provider is reachable."
    ),
    recommendations=[],
)


class LLMTuner:
    """Review trade performance and emit a :class:`TuningPlan`."""

    def __init__(self, provider: LLMProvider, *, timeout_sec: float = TUNER_TIMEOUT_SEC) -> None:
        self._provider = provider
        self._timeout = timeout_sec

    async def generate(
        self,
        performances: list[StrategySymbolPerformance],
        ctx: TuningContext,
    ) -> tuple[TuningPlan, TunerMeta]:
        """Call the LLM and parse a tuning plan (fallback on any error)."""
        prompt = build_tuning_prompt(performances, ctx)
        start = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                self._provider.complete(prompt),
                timeout=self._timeout,
            )
        except (TimeoutError, httpx.HTTPError) as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "tuner_fallback_transport",
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=latency_ms,
            )
            return DEFAULT_TUNING_PLAN, TunerMeta(
                status="fallback_transport",
                provider="fallback",
                latency_ms=latency_ms,
                error=str(exc),
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "tuner_fallback_unexpected",
                error=str(exc),
                latency_ms=latency_ms,
            )
            return DEFAULT_TUNING_PLAN, TunerMeta(
                status="fallback_unexpected",
                provider="fallback_unexpected",
                latency_ms=latency_ms,
                error=str(exc),
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            plan = _parse_plan(raw)
            plan = sanitise_plan(plan)
            return plan, TunerMeta(
                status="ok",
                provider=self._provider.name,
                latency_ms=latency_ms,
                raw_preview=raw[:RAW_PREVIEW_CHARS],
            )
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
            logger.warning(
                "tuner_fallback_parse_error",
                error=str(exc),
                latency_ms=latency_ms,
                raw_preview=raw[:RAW_PREVIEW_CHARS],
            )
            return DEFAULT_TUNING_PLAN, TunerMeta(
                status="fallback_parse_error",
                provider="fallback_parse_error",
                latency_ms=latency_ms,
                error=str(exc),
                raw_preview=raw[:RAW_PREVIEW_CHARS],
            )


def _parse_plan(raw: str) -> TuningPlan:
    payload = extract_json_object(raw)
    data = json.loads(payload)
    if not isinstance(data, dict):
        msg = f"JSON root must be object, got {type(data).__name__}"
        raise ValueError(msg)
    return TuningPlan.model_validate(data)
