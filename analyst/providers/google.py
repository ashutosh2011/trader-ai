"""Google Gemini LLM provider.

Auto-routes between two Google endpoints based on the API key shape:

- ``AIza...`` keys (AI Studio "classic" / Generative Language API) call
  ``generativelanguage.googleapis.com``.
- ``AQ.*`` keys (Vertex AI Express Mode / Agent Platform Studio) call
  ``aiplatform.googleapis.com``. These are issued at
  ``console.cloud.google.com/agent-platform/studio/settings/api-keys``
  and authenticate to Vertex AI without a project ID or location.

The two endpoints accept nearly identical request bodies; the only
substantive difference is that Vertex requires ``role: "user"`` on the
content entries, which the Generative Language API also accepts. We
always send the role to keep the wire format unified.
"""

import json
from typing import Any

import httpx
import structlog

from analyst.provider import LLMProvider
from config.settings import AnalystProviderConfig

logger = structlog.get_logger(__name__)


VERTEX_EXPRESS_BASE = "https://aiplatform.googleapis.com/v1"
GENERATIVE_LANGUAGE_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _is_vertex_express_key(api_key: str) -> bool:
    """Return ``True`` when ``api_key`` looks like a Vertex Express token."""
    return api_key.strip().startswith("AQ.")


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
            "Cloud project, OR use a Vertex AI Express key (AQ.* prefix from "
            "Agent Platform Studio), then retry."
        )
        if activation_url:
            return f"{hint} Open: {activation_url}"
        return hint

    return f"Gemini request failed ({response.status_code}): {message}"


class GoogleProvider(LLMProvider):
    """Async Google Gemini client supporting both AIza and AQ keys.

    Uses Gemini's ``generation_config.response_mime_type = "application/json"``
    to enforce a JSON-only response body. Endpoint base is auto-selected
    from the key shape.
    """

    def __init__(self, config: AnalystProviderConfig) -> None:
        if not config.google_api_key:
            msg = "GOOGLE_API_KEY not configured"
            raise ValueError(msg)
        self._api_key = config.google_api_key
        self._model = config.model_google
        self._uses_vertex_express = _is_vertex_express_key(self._api_key)
        if self._uses_vertex_express:
            self._base_url = VERTEX_EXPRESS_BASE
        else:
            self._base_url = GENERATIVE_LANGUAGE_BASE

    @property
    def name(self) -> str:
        return "google"

    @property
    def backend(self) -> str:
        """Return ``"vertex"`` for AQ.* keys or ``"genlang"`` otherwise."""
        return "vertex" if self._uses_vertex_express else "genlang"

    async def complete(self, prompt: str) -> str:
        """Call Gemini ``generateContent`` and return concatenated text parts.

        Args:
            prompt: User message contents.

        Returns:
            The first candidate's concatenated text parts.
        """
        if self._uses_vertex_express:
            # TRADEOFF: Vertex Express omits the ``models/`` segment;
            # Generative Language nests under ``models/``. Both accept
            # the same request body when ``role: "user"`` is set.
            url = f"{self._base_url}/publishers/google/models/{self._model}:generateContent"
        else:
            url = f"{self._base_url}/models/{self._model}:generateContent"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                params={"key": self._api_key},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
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
