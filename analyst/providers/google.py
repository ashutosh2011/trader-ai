"""Google Gemini LLM provider."""

import httpx
import structlog

from analyst.provider import LLMProvider
from config.settings import AnalystProviderConfig

logger = structlog.get_logger(__name__)


class GoogleProvider(LLMProvider):
    """Async Google Generative Language API client.

    Uses Gemini's ``generation_config.response_mime_type = "application/json"``
    to enforce a JSON-only response body.
    """

    def __init__(self, config: AnalystProviderConfig) -> None:
        if not config.google_api_key:
            msg = "GOOGLE_API_KEY not configured"
            raise ValueError(msg)
        self._api_key = config.google_api_key
        self._model = config.model_google

    @property
    def name(self) -> str:
        return "google"

    async def complete(self, prompt: str) -> str:
        """Call Gemini ``generateContent`` and return concatenated text parts.

        Args:
            prompt: User message contents.

        Returns:
            The first candidate's concatenated text parts.
        """
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                params={"key": self._api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "response_mime_type": "application/json",
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(str(p.get("text", "")) for p in parts)
