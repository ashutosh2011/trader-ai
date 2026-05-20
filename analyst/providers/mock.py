"""Mock LLM provider for deterministic tests."""

from analyst.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Returns a fixed response string."""

    def __init__(self, response: str, *, name: str = "mock") -> None:
        self._response = response
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, prompt: str) -> str:
        return self._response
