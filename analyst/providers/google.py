"""Google Gemini LLM provider."""

import json
from typing import Any

import httpx
import structlog

from analyst.provider import LLMProvider
from config.settings import AnalystProviderConfig

logger = structlog.get_logger(__name__)


def _google_api_error_message(response: httpx.Response) -> str:
    """Turn a Gemini HTTP error body into an operator-friendly message."""
    try:
        payload: dict[str, Any] = response.json()
    except (json.JSONDecodeError, ValueError):
        return f"Gemini request failed ({response.status_code}): {response.text[:300]}"

    err = payload.get("error")
    if not isinstance(err, dict):
        return f"Gemini request failed ({response.status_code})"

    message = str(err.get("message") or "unknown error")
    details = err.get("details")
    activation_url: str | None = None
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata")
            if isinstance(meta, dict) and meta.get("activationUrl"):
                activation_url = str(meta["activationUrl"])
                break

    if activation_url or "has not been used in project" in message.lower():
        hint = (
            "Enable the Generative Language API (Gemini API) for your Google "
            "Cloud project, then retry."
        )
        if activation_url:
            return f"{hint} Open: {activation_url}"
        return hint

    return f"Gemini request failed ({response.status_code}): {message}"


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
            if response.is_error:
                msg = _google_api_error_message(response)
                raise httpx.HTTPStatusError(
                    msg,
                    request=response.request,
                    response=response,
                )
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(str(p.get("text", "")) for p in parts)
