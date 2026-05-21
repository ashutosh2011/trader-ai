"""Signal analyst with LLM advisory and timeout fallback."""

import asyncio
import json
import re
import time
from typing import Any

import httpx
import structlog
from pydantic import ValidationError

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

# Single-level nested object regex; sufficient for our verdict schema which
# has only flat keys plus optional indicator dicts.
_BALANCED_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


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


def _strip_code_fences(text: str) -> str:
    """Strip triple-backtick fences (with optional language tag) if present."""
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    # Drop the opening fence line (may include a language tag like ```json).
    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def _extract_json_object(raw: str) -> str:
    """Extract a JSON object substring from raw LLM text.

    Layered strategy:
        1. Strip surrounding whitespace.
        2. If the text starts with ``{`` and ends with ``}``, return it.
        3. Search for the first balanced ``{...}`` substring (one level of
           nesting supported, sufficient for our schema).
        4. If still not found, strip triple-backtick fences (optionally
           tagged ``json``) and retry the balanced-object search.
        5. Raise :class:`ValueError` if no candidate JSON object is found.

    Args:
        raw: Raw text returned by an LLM provider.

    Returns:
        A JSON object substring suitable for :func:`json.loads`.

    Raises:
        ValueError: If no balanced JSON object can be located.
    """
    text = raw.strip()
    if not text:
        msg = "empty LLM response"
        raise ValueError(msg)
    if text.startswith("{") and text.endswith("}"):
        return text
    match = _BALANCED_OBJECT_RE.search(text)
    if match is not None:
        return match.group(0)
    # Fallback: strip code fences and look again.
    fenced = _strip_code_fences(text)
    if fenced != text:
        if fenced.startswith("{") and fenced.endswith("}"):
            return fenced
        match = _BALANCED_OBJECT_RE.search(fenced)
        if match is not None:
            return match.group(0)
    msg = "no JSON object found in LLM response"
    raise ValueError(msg)


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
