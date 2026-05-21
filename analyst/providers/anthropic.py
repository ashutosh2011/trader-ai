"""Anthropic Claude LLM provider."""

import json

import httpx
import structlog

from analyst.provider import LLMProvider
from analyst.verdict import Verdict, VerdictAction
from config.settings import AnalystProviderConfig

logger = structlog.get_logger(__name__)

# Anthropic's "structured output" without tool-use is achieved by prefilling
# the assistant turn with an opening brace so the model continues a JSON
# object. We re-add the leading '{' before parsing.
_ASSISTANT_PREFILL = "{"


class AnthropicProvider(LLMProvider):
    """Async Anthropic Messages API client with JSON prefill.

    TRADEOFF: Anthropic has no one-shot ``response_format=json`` flag.
    We use an assistant-turn prefill of ``{`` to force the model to
    continue a JSON object. Tool-use mode would be more rigorous but
    requires schema declaration plus tool-result handling on every call;
    prefill is lighter weight for our flat verdict schema.
    """

    def __init__(self, config: AnalystProviderConfig) -> None:
        if not config.anthropic_api_key:
            msg = "ANTHROPIC_API_KEY not configured"
            raise ValueError(msg)
        self._api_key = config.anthropic_api_key
        self._model = config.model_anthropic

    @property
    def name(self) -> str:
        return "anthropic"

    async def complete(self, prompt: str) -> str:
        """Call Anthropic messages API and return JSON text.

        Args:
            prompt: User message contents.

        Returns:
            Concatenated text content blocks, prefixed with the assistant
            prefill (``{``) so the response is a parseable JSON object.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 256,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": _ASSISTANT_PREFILL},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()
            blocks = data.get("content", [])
            texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            body = "".join(texts)
            # Re-attach the prefill so downstream parser sees a complete object.
            if body.lstrip().startswith(_ASSISTANT_PREFILL):
                return body
            return _ASSISTANT_PREFILL + body


def parse_verdict_json(raw: str, *, provider: str, latency_ms: int) -> Verdict:
    """Parse LLM JSON text into a :class:`Verdict`.

    Retained for backward compatibility with earlier tests; new code should
    use :func:`analyst.analyst._parse_verdict` via :meth:`Analyst.analyze`.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(text)
    action: VerdictAction = data["action"]
    multiplier = min(1.0, float(data.get("size_multiplier", 1.0)))
    return Verdict(
        action=action,
        size_multiplier=multiplier,
        confidence=float(data.get("confidence", 0.5)),
        rationale=str(data.get("rationale", "")),
        provider=provider,
        latency_ms=latency_ms,
    )
