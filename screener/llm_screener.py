"""LLM-facing screener layer with fallback ladder.

Mirrors :class:`analyst.analyst.Analyst` in structure: a single
``generate`` method runs the provider under a hard timeout, parses the
response, and falls back to :data:`DEFAULT_FORMULA` on any error. Unlike
the analyst, the fallback **never vetoes** — the worst case is producing
a conservative default formula that the runner still evaluates.

TRADEOFF: We use an 8s timeout vs the analyst's 2s because the screener
is not in the trading hot path. It runs on-demand from the dashboard or
CLI, and the LLM can think harder about regime selection. Documented
inline at :data:`SCREENER_TIMEOUT_SEC`.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from analyst._parsing import extract_json_object
from analyst.provider import LLMProvider
from screener.prompt import MarketContext, build_screener_prompt
from screener.schema import (
    CompareTo,
    IndicatorFilter,
    ScreenerFormula,
)
from screener.universe import Universe

logger = structlog.get_logger(__name__)

SCREENER_TIMEOUT_SEC = 8.0
"""Wall-clock timeout for one LLM screener generation.

Larger than the analyst's 2s because the screener runs on-demand (not in
the per-signal hot path) and benefits from giving the model more headroom
to reason about regime. Tune via constructor argument if needed.
"""

RAW_PREVIEW_CHARS = 300

ScreenerMetaStatus = Literal[
    "ok",
    "fallback_transport",
    "fallback_parse_error",
    "fallback_unexpected",
]


class ScreenerMeta(BaseModel):
    """Bookkeeping for one screener generation call.

    Persisted on every run record so the dashboard can show provider,
    latency, and the raw preview when something went sideways.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ScreenerMetaStatus
    provider: str
    latency_ms: int = Field(ge=0)
    error: str | None = None
    raw_preview: str | None = None


DEFAULT_FORMULA: ScreenerFormula = ScreenerFormula(
    name="Default fallback: oversold mean-revert",
    timeframe="day",
    side_bias="long",
    rationale="LLM unavailable; using conservative default oversold filter.",
    filters=[
        IndicatorFilter(indicator="rsi", params={"period": 14}, op="<", value=35.0),
        IndicatorFilter(
            indicator="close",
            op=">",
            compare_to=CompareTo(indicator="sma", params={"period": 50}),
        ),
    ],
)


class LLMScreener:
    """Generate a structured :class:`ScreenerFormula` from an LLM provider.

    Outcomes mirror the analyst fallback ladder:
        * Success → ``status="ok"`` with the parsed formula.
        * Timeout / :class:`httpx.HTTPError` → DEFAULT_FORMULA with
          ``status="fallback_transport"``.
        * JSON / schema error → DEFAULT_FORMULA with
          ``status="fallback_parse_error"``.
        * Unexpected exception → DEFAULT_FORMULA with
          ``status="fallback_unexpected"``.

    The screener never raises; the worst case is the default formula
    with a status that the dashboard renders as a yellow / red badge.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        timeout_sec: float = SCREENER_TIMEOUT_SEC,
    ) -> None:
        self._provider = provider
        self._timeout_sec = timeout_sec

    async def generate(
        self,
        ctx: MarketContext,
        universe: Universe,
    ) -> tuple[ScreenerFormula, ScreenerMeta]:
        """Generate a formula. Always returns a tuple — never raises."""
        prompt = build_screener_prompt(ctx, universe)
        start = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                self._provider.complete(prompt),
                timeout=self._timeout_sec,
            )
        except (TimeoutError, httpx.HTTPError) as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "screener_fallback_transport",
                error=str(exc),
                error_type=type(exc).__name__,
                provider=self._provider.name,
                latency_ms=latency_ms,
            )
            return DEFAULT_FORMULA, ScreenerMeta(
                status="fallback_transport",
                provider=self._provider.name,
                latency_ms=latency_ms,
                error=str(exc),
                raw_preview=None,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "screener_fallback_unexpected",
                error=str(exc),
                error_type=type(exc).__name__,
                provider=self._provider.name,
                latency_ms=latency_ms,
            )
            return DEFAULT_FORMULA, ScreenerMeta(
                status="fallback_unexpected",
                provider=self._provider.name,
                latency_ms=latency_ms,
                error=str(exc),
                raw_preview=None,
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            formula = _parse_formula(raw)
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
            logger.warning(
                "screener_fallback_parse_error",
                error=str(exc),
                error_type=type(exc).__name__,
                provider=self._provider.name,
                latency_ms=latency_ms,
                raw_preview=raw[:RAW_PREVIEW_CHARS],
            )
            return DEFAULT_FORMULA, ScreenerMeta(
                status="fallback_parse_error",
                provider=self._provider.name,
                latency_ms=latency_ms,
                error=str(exc),
                raw_preview=raw[:RAW_PREVIEW_CHARS],
            )
        return formula, ScreenerMeta(
            status="ok",
            provider=self._provider.name,
            latency_ms=latency_ms,
            error=None,
            raw_preview=raw[:RAW_PREVIEW_CHARS],
        )


def _parse_formula(raw: str) -> ScreenerFormula:
    """Extract and validate a :class:`ScreenerFormula` from raw text.

    Args:
        raw: LLM provider response.

    Returns:
        Validated formula.

    Raises:
        json.JSONDecodeError: When the extracted object isn't valid JSON.
        ValueError: When no JSON object can be located or the parsed
            value isn't a mapping.
        pydantic.ValidationError: When the parsed dict fails the
            :class:`ScreenerFormula` schema.
    """
    payload = extract_json_object(raw)
    data: Any = json.loads(payload)
    if not isinstance(data, dict):
        msg = f"JSON root must be an object, got {type(data).__name__}"
        raise ValueError(msg)
    return ScreenerFormula.model_validate(data)


__all__ = [
    "DEFAULT_FORMULA",
    "LLMScreener",
    "SCREENER_TIMEOUT_SEC",
    "ScreenerMeta",
    "ScreenerMetaStatus",
]
