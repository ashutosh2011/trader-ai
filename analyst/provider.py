"""LLM provider abstract base."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Async LLM completion provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier for logging and verdict attribution."""

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Return raw model text (expected JSON for analyst parsing)."""
