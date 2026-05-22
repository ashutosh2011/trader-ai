"""Signal analyst with LLM advisory and timeout fallback."""

import asyncio
import json
import time
from typing import Any

import httpx
import structlog
from pydantic import ValidationError

from analyst._parsing import extract_json_object, strip_code_fences
from analyst.prompt import build_analyst_prompt
from analyst.provider import LLMProvider
from analyst.verdict import Verdict, VerdictAction
from core.context import Context
from core.signal import Signal

logger = structlog.get_logger(__name__)

ANALYST_TIMEOUT_SEC = 2.0
FALLBACK_MULTIPLIER = 0.7
DEFAULT_SIZE_MULTIPLIER = 0.7

VALID_ACTIONS: frozenset[str] = frozenset({"APPROVE", "VETO", "SHRINK"})

# Re-exported for backward compatibility — existing tests import the
# private helpers from this module directly.
_strip_code_fences = strip_code_fences
_extract_json_object = extract_json_object


class Analyst:
    """Advisory analyst: APPROVE, VETO, or SHRINK with size multiplier."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def analyze(self, signal: Signal, ctx: Context) -> Verdict:
        """Analyze a signal with a hard 2s timeout.

        Outcomes:
            * Success: parsed :class:`Verdict` from the provider.
            * Timeout or transport/network error: ``APPROVE`` at
              :data:`FALLBACK_MULTIPLIER` with ``provider="fallback"`` —
              the provider was unreachable but the rules-only signal
              still stands at reduced size.
            * Parse error (malformed JSON, missing keys, schema violation):
              ``VETO`` at 0.0 with ``provider="fallback_parse_error"`` —
              we refuse to trade on garbage LLM output.
            * Any other unexpected exception: ``VETO`` at 0.0 with
              ``provider="fallback_unexpected"`` (safe default).

        Args:
            signal: Strategy signal to advise on.
            ctx: Strategy context for the current bar.

        Returns:
            A :class:`Verdict` describing the advisory action.
        """
        prompt = build_analyst_prompt(signal, ctx)
        start = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                self._provider.complete(prompt),
                timeout=ANALYST_TIMEOUT_SEC,
            )
        except (TimeoutError, httpx.HTTPError) as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "analyst_fallback_transport",
                error=str(exc),
                error_type=type(exc).__name__,
                symbol=signal.symbol,
                latency_ms=latency_ms,
            )
            return Verdict(
                action="APPROVE",
                size_multiplier=FALLBACK_MULTIPLIER,
                confidence=0.5,
                rationale=f"fallback due to transport error: {exc}",
                provider="fallback",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "analyst_fallback_unexpected",
                error=str(exc),
                error_type=type(exc).__name__,
                symbol=signal.symbol,
                latency_ms=latency_ms,
            )
            return Verdict(
                action="VETO",
                size_multiplier=0.0,
                confidence=0.0,
                rationale=f"veto due to unexpected error: {exc}",
                provider="fallback_unexpected",
                latency_ms=latency_ms,
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            return _parse_verdict(
                raw,
                provider=self._provider.name,
                latency_ms=latency_ms,
            )
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
            logger.warning(
                "analyst_fallback_parse_error",
                error=str(exc),
                error_type=type(exc).__name__,
                symbol=signal.symbol,
                latency_ms=latency_ms,
                raw_preview=raw[:200],
            )
            return Verdict(
                action="VETO",
                size_multiplier=0.0,
                confidence=0.0,
                rationale=f"veto due to parse error: {exc}",
                provider="fallback_parse_error",
                latency_ms=latency_ms,
            )


def _parse_verdict(raw: str, *, provider: str, latency_ms: int) -> Verdict:
    """Parse raw LLM text into a :class:`Verdict`.

    Args:
        raw: Raw model output (may include commentary or code fences).
        provider: Provider name to attribute the verdict to.
        latency_ms: Observed provider latency in milliseconds.

    Returns:
        The parsed :class:`Verdict`.

    Raises:
        json.JSONDecodeError: If the extracted object is not valid JSON.
        KeyError: If the required ``action`` key is missing.
        ValueError: If the response is empty/garbage, contains no JSON
            object, or the action is outside the allowed enum.
        pydantic.ValidationError: If the constructed model is invalid.
    """
    payload = _extract_json_object(raw)
    data: Any = json.loads(payload)
    if not isinstance(data, dict):
        msg = f"JSON root must be an object, got {type(data).__name__}"
        raise ValueError(msg)
    action_raw = data["action"]
    if not isinstance(action_raw, str) or action_raw not in VALID_ACTIONS:
        msg = f"invalid action {action_raw!r}; expected one of {sorted(VALID_ACTIONS)}"
        raise ValueError(msg)
    action: VerdictAction = action_raw  # type: ignore[assignment]
    # TRADEOFF: missing size_multiplier defaults to DEFAULT_SIZE_MULTIPLIER
    # (0.7), matching the timeout-fallback size. An LLM that omits the field
    # should not auto-promote to full size; clamp to [0, 1].
    raw_multiplier = data.get("size_multiplier", DEFAULT_SIZE_MULTIPLIER)
    multiplier = min(1.0, max(0.0, float(raw_multiplier)))
    confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
    return Verdict(
        action=action,
        size_multiplier=multiplier,
        confidence=confidence,
        rationale=str(data.get("rationale", "")),
        provider=provider,
        latency_ms=latency_ms,
    )
