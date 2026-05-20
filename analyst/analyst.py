"""Signal analyst with LLM advisory and timeout fallback."""

import asyncio
import json
import time

import structlog

from analyst.prompt import build_analyst_prompt
from analyst.provider import LLMProvider
from analyst.verdict import Verdict, VerdictAction
from core.context import Context
from core.signal import Signal

logger = structlog.get_logger(__name__)

ANALYST_TIMEOUT_SEC = 2.0
FALLBACK_MULTIPLIER = 0.7


class Analyst:
    """Advisory analyst: APPROVE, VETO, or SHRINK with size multiplier."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def analyze(self, signal: Signal, ctx: Context) -> Verdict:
        """Analyze a signal with a hard 2s timeout.

        On timeout or error, returns APPROVE at 0.7x with provider ``fallback``.
        """
        prompt = build_analyst_prompt(signal, ctx)
        start = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                self._provider.complete(prompt),
                timeout=ANALYST_TIMEOUT_SEC,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            return _parse_verdict(
                raw,
                provider=self._provider.name,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "analyst_fallback",
                error=str(exc),
                symbol=signal.symbol,
                latency_ms=latency_ms,
            )
            return Verdict(
                action="APPROVE",
                size_multiplier=FALLBACK_MULTIPLIER,
                confidence=0.5,
                rationale=f"fallback due to error: {exc}",
                provider="fallback",
                latency_ms=latency_ms,
            )


def _parse_verdict(raw: str, *, provider: str, latency_ms: int) -> Verdict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    action: VerdictAction = data["action"]
    multiplier = min(1.0, max(0.0, float(data.get("size_multiplier", 1.0))))
    return Verdict(
        action=action,
        size_multiplier=multiplier,
        confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
        rationale=str(data.get("rationale", "")),
        provider=provider,
        latency_ms=latency_ms,
    )
