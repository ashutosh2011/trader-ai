"""Bridge between dashboard routes/CLI and :class:`ScreenerRunner`.

Builds the LLM provider, candle store, optional Kite fetcher, and the
:class:`ScreenerRunner` from an :class:`AppState` snapshot. Also owns
the deterministic stub provider used by the dashboard "provider=stub"
option and by tests.

TRADEOFF: The service constructs a *fresh* :class:`CandleStore` per
run so we don't compete with the live loop's connection. The store is
closed in the finally block. This matches the backtest runner's pattern
for the same reason.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import structlog

from analyst.provider import LLMProvider
from analyst.providers.anthropic import AnthropicProvider
from analyst.providers.google import GoogleProvider
from analyst.providers.mock import MockLLMProvider
from analyst.providers.openai import OpenAIProvider
from config.settings import AppSettings
from data.historical import HistoricalFetcher
from data.kite_client import KiteClient
from data.store import CandleStore
from screener.llm_screener import DEFAULT_FORMULA, LLMScreener
from screener.prompt import MarketContext
from screener.runner import ScreenerRunner, ScreenerRunRecord
from screener.store import ScreenerStore
from screener.universe import Universe, load_universe

logger = structlog.get_logger(__name__)

ScreenerProviderName = Literal["openai", "anthropic", "google", "stub"]

PROVIDER_OPTIONS: tuple[ScreenerProviderName, ...] = (
    "stub",
    "openai",
    "anthropic",
    "google",
)


@dataclass(frozen=True)
class _StubResponses:
    """Deterministic LLM responses keyed by provider name."""

    default_formula_json: str


_STUB_DEFAULT_FORMULA_JSON: str = DEFAULT_FORMULA.model_dump_json()
_STUB_RESPONSES = _StubResponses(default_formula_json=_STUB_DEFAULT_FORMULA_JSON)


KiteClientFactory = Callable[[AppSettings], KiteClient]


def build_stub_provider() -> LLMProvider:
    """Return a deterministic provider that always returns DEFAULT_FORMULA."""
    return MockLLMProvider(_STUB_RESPONSES.default_formula_json, name="stub")


def build_provider(
    name: ScreenerProviderName,
    settings: AppSettings,
) -> LLMProvider:
    """Construct an LLM provider for the dashboard / CLI.

    Args:
        name: One of :data:`PROVIDER_OPTIONS`.
        settings: Source of API keys.

    Returns:
        An :class:`LLMProvider` ready to call.

    Raises:
        ValueError: If the chosen provider is missing credentials.
    """
    if name == "stub":
        return build_stub_provider()
    if name == "anthropic":
        return AnthropicProvider(settings.analyst)
    if name == "openai":
        return OpenAIProvider(settings.analyst)
    if name == "google":
        return GoogleProvider(settings.analyst)
    msg = f"unknown screener provider: {name}"  # pragma: no cover - typed Literal
    raise ValueError(msg)


class ScreenerService:
    """High-level facade over the screener stack for dashboard / CLI use."""

    def __init__(
        self,
        store: ScreenerStore,
        settings: AppSettings,
        *,
        kite_client_factory: KiteClientFactory | None = None,
        candle_store_factory: Callable[[AppSettings], CandleStore] | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._kite_client_factory = kite_client_factory or KiteClient.from_settings
        self._candle_store_factory = candle_store_factory or _default_candle_store

    def load_universe(self) -> Universe:
        """Load the configured screener universe."""
        return load_universe()

    async def run(
        self,
        *,
        provider_name: ScreenerProviderName,
        market_context: MarketContext,
        fetch_missing: bool = False,
        bars_back: int = 200,
        universe: Universe | None = None,
        run_id: str | None = None,
    ) -> ScreenerRunRecord:
        """Run a screener end-to-end.

        Raises:
            ValueError: If the provider requires credentials that are not
                configured, or if ``fetch_missing=True`` is requested
                without Kite credentials.
        """
        provider = build_provider(provider_name, self._settings)
        loaded_universe = universe if universe is not None else self.load_universe()
        llm = LLMScreener(provider)
        candle_store = self._candle_store_factory(self._settings)
        fetcher: HistoricalFetcher | None = None
        if fetch_missing:
            if not self._settings.kite_configured():
                candle_store.close()
                msg = (
                    "fetch_missing=True requires KITE_API_KEY and KITE_ACCESS_TOKEN; "
                    "configure Kite or rerun with fetch_missing=False"
                )
                raise ValueError(msg)
            kite = self._kite_client_factory(self._settings)
            fetcher = HistoricalFetcher(kite, candle_store)
        runner = ScreenerRunner(
            llm_screener=llm,
            candle_store=candle_store,
            store=self._store,
            fetcher=fetcher,
        )
        try:
            return await runner.run(
                loaded_universe,
                market_context,
                run_id=run_id,
                fetch_missing=fetch_missing,
                bars_back=bars_back,
            )
        finally:
            candle_store.close()


def filter_to_sentence(filter_payload: dict[str, object]) -> str:
    """Render a filter dict into a human-readable sentence.

    The dashboard detail page calls this for every filter so the LLM's
    formula reads naturally next to the raw JSON.
    """
    filter_type = filter_payload.get("type")
    op = str(filter_payload.get("op", "?"))
    if filter_type == "indicator":
        indicator = str(filter_payload.get("indicator", "?"))
        params_raw = filter_payload.get("params") or {}
        params = _coerce_params_dict(params_raw)
        lhs = _indicator_label(indicator, params)
        compare_to = filter_payload.get("compare_to")
        if isinstance(compare_to, dict):
            rhs = _indicator_label(
                str(compare_to.get("indicator", "?")),
                _coerce_params_dict(compare_to.get("params") or {}),
            )
        else:
            value = filter_payload.get("value")
            rhs = f"{float(value):g}" if isinstance(value, int | float) else str(value)
        return f"{lhs} {op} {rhs}"
    if filter_type == "volume":
        value_x_avg = filter_payload.get("value_x_avg")
        avg_window = filter_payload.get("avg_window", 20)
        if isinstance(value_x_avg, int | float):
            return f"volume {op} {float(value_x_avg):g}× avg({avg_window})"
        value = filter_payload.get("value")
        return f"volume {op} {value}"
    if filter_type == "price_change":
        window = filter_payload.get("window", "?")
        value_pct = filter_payload.get("value_pct", "?")
        return (
            f"price change over {window} bars {op} {value_pct}%"
            if isinstance(value_pct, int | float)
            else f"price change over {window} bars {op} {value_pct}"
        )
    return json.dumps(filter_payload)


def _coerce_params_dict(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, val in raw.items():
        if isinstance(val, int | float):
            out[str(key)] = float(val)
    return out


def _indicator_label(name: str, params: dict[str, float]) -> str:
    if not params:
        return name.upper() if name in {"rsi", "sma", "ema", "atr"} else name
    parts = ",".join(
        f"{k}={int(v) if float(v).is_integer() else v}" for k, v in params.items()
    )
    display = name.upper() if name in {"rsi", "sma", "ema", "atr"} else name
    return f"{display}({parts})"


def _default_candle_store(settings: AppSettings) -> CandleStore:
    return CandleStore(settings.data.duckdb_path)


__all__ = [
    "PROVIDER_OPTIONS",
    "ScreenerProviderName",
    "ScreenerService",
    "build_provider",
    "build_stub_provider",
    "filter_to_sentence",
]
