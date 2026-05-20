"""OpenAI LLM provider."""

import httpx
import structlog

from analyst.provider import LLMProvider
from config.settings import AnalystProviderConfig

logger = structlog.get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """Async OpenAI Chat Completions API client."""

    def __init__(self, config: AnalystProviderConfig) -> None:
        if not config.openai_api_key:
            msg = "OPENAI_API_KEY not configured"
            raise ValueError(msg)
        self._api_key = config.openai_api_key
        self._model = config.model_openai

    @property
    def name(self) -> str:
        return "openai"

    async def complete(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                },
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
