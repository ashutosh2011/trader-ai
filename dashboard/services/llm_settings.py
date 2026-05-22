"""Read / write LLM provider settings from the dashboard.

The ``/llm`` page lets the operator inspect which provider keys are
configured (masked), drop in new keys, pick a default model per
provider, and run a tiny connectivity ping. Keys land in ``.env``
(gitignored); models land in ``config/config.yaml`` under the
``analyst:`` block — preserving every other YAML key.

TRADEOFF: We never echo the raw key back to the UI. The page renders
only a 6+4 char preview. The ``test_connection`` method builds a real
provider through an injectable factory so tests can stub the network
call without losing coverage of the surrounding orchestration.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import structlog
import yaml  # type: ignore[import-untyped]

from analyst.provider import LLMProvider
from analyst.providers.anthropic import AnthropicProvider
from analyst.providers.google import GoogleProvider
from analyst.providers.openai import OpenAIProvider
from config.settings import AnalystProviderConfig, AppSettings
from dashboard.services.env_writer import delete_env_var, upsert_env_var

logger = structlog.get_logger(__name__)


ProviderName = Literal["anthropic", "openai", "google"]
DefaultProviderName = Literal["anthropic", "openai", "google", "mock"]
PROVIDER_NAMES: tuple[ProviderName, ...] = ("anthropic", "openai", "google")

ENV_KEYS: dict[ProviderName, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}

MODEL_OPTIONS: dict[ProviderName, tuple[str, ...]] = {
    "anthropic": (
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-opus-latest",
    ),
    "openai": ("gpt-4o-mini", "gpt-4o", "gpt-4-turbo"),
    "google": ("gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"),
}

ProviderFactory = Callable[[ProviderName, AnalystProviderConfig], LLMProvider]


def _default_provider_factory(
    name: ProviderName,
    config: AnalystProviderConfig,
) -> LLMProvider:
    """Build the real provider for ``name``; raises if the key is missing."""
    if name == "anthropic":
        return AnthropicProvider(config)
    if name == "openai":
        return OpenAIProvider(config)
    if name == "google":
        return GoogleProvider(config)
    msg = f"unknown provider: {name}"
    raise ValueError(msg)


@dataclass(frozen=True)
class ProviderStatus:
    """One provider's masked status (for the page card)."""

    name: ProviderName
    env_key: str
    configured: bool
    preview: str
    model: str
    model_options: list[str]


@dataclass(frozen=True)
class LLMSettingsSnapshot:
    """Aggregate read-state shown on the ``/llm`` page."""

    providers: list[ProviderStatus]
    default_provider: DefaultProviderName
    env_path: Path
    config_path: Path


@dataclass(frozen=True)
class LLMTestResult:
    """Outcome of a tiny provider ping."""

    ok: bool
    latency_ms: int
    error: str | None
    response_preview: str | None


def _mask_key(value: str | None) -> str:
    """Render a 6+4 preview, or empty string when no key is set."""
    if not value:
        return ""
    text = value.strip()
    if not text:
        return ""
    if len(text) <= 10:
        return text[0] + "…" + text[-1]
    return f"{text[:6]}…{text[-4:]}"


class LLMSettingsService:
    """Glue between :class:`AppSettings`, ``.env`` and ``config.yaml``."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        env_path: Path,
        config_path: Path,
        provider_factory: ProviderFactory | None = None,
        reload_settings: Callable[[], AppSettings] | None = None,
    ) -> None:
        """Bind the service to a settings snapshot + on-disk paths.

        Args:
            settings: Current :class:`AppSettings`.
            env_path: Target ``.env`` path for key writes.
            config_path: ``config.yaml`` path for model writes.
            provider_factory: Override the real provider constructor —
                tests inject a stub that returns :class:`MockLLMProvider`.
            reload_settings: Called after writes so the in-memory
                snapshot reflects the new key. Defaults to a no-op.
        """
        self._settings = settings
        self._env_path = env_path
        self._config_path = config_path
        self._provider_factory = provider_factory or _default_provider_factory
        self._reload = reload_settings or (lambda: settings)

    @property
    def settings(self) -> AppSettings:
        """Latest :class:`AppSettings` snapshot the service is bound to."""
        return self._settings

    def read_status(self) -> LLMSettingsSnapshot:
        """Build a masked snapshot for the ``/llm`` page."""
        analyst = self._settings.analyst
        providers: list[ProviderStatus] = []
        key_lookup: dict[ProviderName, str | None] = {
            "anthropic": analyst.anthropic_api_key,
            "openai": analyst.openai_api_key,
            "google": analyst.google_api_key,
        }
        model_lookup: dict[ProviderName, str] = {
            "anthropic": analyst.model_anthropic,
            "openai": analyst.model_openai,
            "google": analyst.model_google,
        }
        for name in PROVIDER_NAMES:
            raw = key_lookup[name]
            providers.append(
                ProviderStatus(
                    name=name,
                    env_key=ENV_KEYS[name],
                    configured=bool(raw),
                    preview=_mask_key(raw),
                    model=model_lookup[name],
                    model_options=list(MODEL_OPTIONS[name]),
                )
            )
        return LLMSettingsSnapshot(
            providers=providers,
            default_provider=analyst.default_provider,
            env_path=self._env_path,
            config_path=self._config_path,
        )

    def update_api_keys(
        self,
        *,
        anthropic: str | None = None,
        openai: str | None = None,
        google: str | None = None,
        delete_anthropic: bool = False,
        delete_openai: bool = False,
        delete_google: bool = False,
    ) -> dict[ProviderName, str]:
        """Write any non-empty key to ``.env``; honour ``delete_*`` flags.

        Args:
            anthropic: New Anthropic key (``None`` or empty leaves the
                existing value alone).
            openai: Same as ``anthropic`` for the OpenAI key.
            google: Same as ``anthropic`` for the Google key.
            delete_anthropic: When ``True``, drop the env line entirely
                even when ``anthropic`` is also provided.
            delete_openai: Same shape for OpenAI.
            delete_google: Same shape for Google.

        Returns:
            A ``{provider: preview}`` dict of the resulting masked
            previews after the writes.
        """
        updates: dict[ProviderName, tuple[str | None, bool]] = {
            "anthropic": (anthropic, delete_anthropic),
            "openai": (openai, delete_openai),
            "google": (google, delete_google),
        }
        for provider, (value, delete) in updates.items():
            env_key = ENV_KEYS[provider]
            if delete:
                removed = delete_env_var(self._env_path, env_key)
                logger.info(
                    "dashboard_llm_key_deleted",
                    provider=provider,
                    removed=removed,
                )
                continue
            if value is None:
                continue
            stripped = value.strip()
            if stripped == "":
                continue
            upsert_env_var(self._env_path, env_key, stripped)
            logger.info(
                "dashboard_llm_key_written",
                provider=provider,
                preview=_mask_key(stripped),
            )
        self._settings = self._reload()
        snapshot = self.read_status()
        return {p.name: p.preview for p in snapshot.providers}

    def update_models(
        self,
        *,
        model_anthropic: str | None = None,
        model_openai: str | None = None,
        model_google: str | None = None,
        default_provider: DefaultProviderName | None = None,
    ) -> None:
        """Merge model / default-provider updates into ``config.yaml``.

        Only the ``analyst`` block is touched; every other key in the
        file is preserved. The validated YAML is reloaded into the
        :class:`AppSettings` snapshot.

        Args:
            model_anthropic: New default Claude model id (or ``None``).
            model_openai: New default OpenAI chat model id (or ``None``).
            model_google: New default Gemini model id (or ``None``).
            default_provider: New analyst-layer default provider.
        """
        updates: dict[str, str] = {}
        if model_anthropic is not None:
            self._validate_model("anthropic", model_anthropic)
            updates["model_anthropic"] = model_anthropic
        if model_openai is not None:
            self._validate_model("openai", model_openai)
            updates["model_openai"] = model_openai
        if model_google is not None:
            self._validate_model("google", model_google)
            updates["model_google"] = model_google
        if default_provider is not None:
            if default_provider not in {"anthropic", "openai", "google", "mock"}:
                msg = f"unknown default_provider: {default_provider}"
                raise ValueError(msg)
            updates["default_provider"] = default_provider
        if not updates:
            return
        _merge_yaml_block(self._config_path, "analyst", updates)
        self._settings = self._reload()
        logger.info(
            "dashboard_llm_models_saved",
            updates=sorted(updates.keys()),
            config_path=str(self._config_path),
        )

    async def test_connection(
        self,
        provider_name: ProviderName,
        *,
        timeout_s: float = 5.0,
        prompt: str = "Reply with exactly: PONG",
    ) -> LLMTestResult:
        """Build the provider and run a tiny completion ping.

        Returns a :class:`LLMTestResult`; failures (missing key,
        transport errors, timeouts) are caught and reported via
        ``error`` so the UI can show the message inline.
        """
        analyst = self._settings.analyst
        try:
            provider = self._provider_factory(provider_name, analyst)
        except ValueError as exc:
            return LLMTestResult(
                ok=False,
                latency_ms=0,
                error=str(exc),
                response_preview=None,
            )
        start = time.perf_counter()
        try:
            text = await asyncio.wait_for(provider.complete(prompt), timeout=timeout_s)
        except TimeoutError:
            return LLMTestResult(
                ok=False,
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"timeout after {timeout_s:.1f}s",
                response_preview=None,
            )
        except Exception as exc:  # pragma: no cover - exercised via mocks
            return LLMTestResult(
                ok=False,
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=f"{type(exc).__name__}: {exc}",
                response_preview=None,
            )
        latency_ms = int((time.perf_counter() - start) * 1000)
        preview = text.strip()
        if len(preview) > 80:
            preview = preview[:77] + "..."
        return LLMTestResult(
            ok=True,
            latency_ms=latency_ms,
            error=None,
            response_preview=preview,
        )

    @staticmethod
    def _validate_model(provider: ProviderName, model: str) -> None:
        text = model.strip()
        if not text:
            msg = f"model_{provider} must not be empty"
            raise ValueError(msg)


def _merge_yaml_block(config_path: Path, block: str, updates: dict[str, Any]) -> None:
    """Read ``config_path``, merge ``updates`` into ``block``, write back.

    Creates the file (and any missing parent directories) when it does
    not yet exist. Other top-level keys are preserved.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if config_path.is_file():
        raw_obj: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(raw_obj, dict):
            data = cast(dict[str, Any], raw_obj)
    raw_block_obj: Any = data.get(block, {})
    block_value: dict[str, Any] = (
        cast(dict[str, Any], raw_block_obj) if isinstance(raw_block_obj, dict) else {}
    )
    block_value.update(updates)
    data[block] = block_value
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


__all__ = [
    "DefaultProviderName",
    "ENV_KEYS",
    "LLMSettingsService",
    "LLMSettingsSnapshot",
    "LLMTestResult",
    "MODEL_OPTIONS",
    "PROVIDER_NAMES",
    "ProviderFactory",
    "ProviderName",
    "ProviderStatus",
]
