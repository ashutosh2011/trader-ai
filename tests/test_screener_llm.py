"""Tests for the LLM screener fallback ladder."""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from analyst.provider import LLMProvider
from analyst.providers.mock import MockLLMProvider
from screener.llm_screener import (
    DEFAULT_FORMULA,
    SCREENER_TIMEOUT_SEC,
    LLMScreener,
)
from screener.prompt import MarketContext
from screener.universe import Universe, UniverseSymbol

IST = ZoneInfo("Asia/Kolkata")


def _ctx() -> MarketContext:
    return MarketContext(
        as_of=datetime(2024, 1, 1, 10, 0, tzinfo=IST),
        recent_index_summary="NIFTY flat",
        notes="",
    )


def _universe() -> Universe:
    return Universe(
        symbols=[
            UniverseSymbol(symbol="A"),
            UniverseSymbol(symbol="B"),
        ]
    )


_VALID_FORMULA_JSON = (
    '{"name": "RSI < 30", "timeframe": "day", "side_bias": "long", '
    '"rationale": "oversold", "filters": ['
    '{"type": "indicator", "indicator": "rsi", "params": {"period": 14}, '
    '"op": "<", "value": 30.0}'
    "]}"
)


@pytest.mark.asyncio
async def test_valid_json_returns_ok() -> None:
    provider = MockLLMProvider(_VALID_FORMULA_JSON, name="mock")
    formula, meta = await LLMScreener(provider).generate(_ctx(), _universe())
    assert formula.name == "RSI < 30"
    assert formula.filters[0].type == "indicator"
    assert meta.status == "ok"
    assert meta.provider == "mock"
    assert meta.error is None
    assert meta.raw_preview is not None


@pytest.mark.asyncio
async def test_garbage_returns_parse_error_fallback() -> None:
    provider = MockLLMProvider("not json", name="mock")
    formula, meta = await LLMScreener(provider).generate(_ctx(), _universe())
    assert formula == DEFAULT_FORMULA
    assert meta.status == "fallback_parse_error"
    assert meta.error is not None
    assert meta.raw_preview is not None


@pytest.mark.asyncio
async def test_schema_violation_returns_parse_error_fallback() -> None:
    # Valid JSON but missing required keys.
    provider = MockLLMProvider('{"name": "bad"}', name="mock")
    formula, meta = await LLMScreener(provider).generate(_ctx(), _universe())
    assert formula == DEFAULT_FORMULA
    assert meta.status == "fallback_parse_error"


@pytest.mark.asyncio
async def test_timeout_returns_transport_fallback() -> None:
    class SlowProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "slow"

        async def complete(self, prompt: str) -> str:
            await asyncio.sleep(SCREENER_TIMEOUT_SEC + 0.5)
            return "{}"

    # Use a very short timeout to keep the test snappy.
    screener = LLMScreener(SlowProvider(), timeout_sec=0.05)
    formula, meta = await screener.generate(_ctx(), _universe())
    assert formula == DEFAULT_FORMULA
    assert meta.status == "fallback_transport"


@pytest.mark.asyncio
async def test_httpx_error_returns_transport_fallback() -> None:
    class HttpxBoom(LLMProvider):
        @property
        def name(self) -> str:
            return "boom"

        async def complete(self, prompt: str) -> str:
            raise httpx.ConnectError("network down")

    formula, meta = await LLMScreener(HttpxBoom()).generate(_ctx(), _universe())
    assert formula == DEFAULT_FORMULA
    assert meta.status == "fallback_transport"
    assert "network down" in (meta.error or "")


@pytest.mark.asyncio
async def test_unexpected_exception_returns_unexpected_fallback() -> None:
    class Broken(LLMProvider):
        @property
        def name(self) -> str:
            return "broken"

        async def complete(self, prompt: str) -> str:
            raise RuntimeError("kaboom")

    formula, meta = await LLMScreener(Broken()).generate(_ctx(), _universe())
    assert formula == DEFAULT_FORMULA
    assert meta.status == "fallback_unexpected"
    assert "kaboom" in (meta.error or "")


@pytest.mark.asyncio
async def test_code_fence_wrapped_json_is_parsed() -> None:
    wrapped = "```json\n" + _VALID_FORMULA_JSON + "\n```"
    provider = MockLLMProvider(wrapped, name="mock")
    formula, meta = await LLMScreener(provider).generate(_ctx(), _universe())
    assert meta.status == "ok"
    assert formula.name == "RSI < 30"


@pytest.mark.asyncio
async def test_default_formula_is_valid() -> None:
    # Sanity: the default formula must be schema-valid (used as fallback).
    assert DEFAULT_FORMULA.timeframe == "day"
    assert DEFAULT_FORMULA.filters
    # And re-serializable / re-parsable.
    blob = DEFAULT_FORMULA.model_dump_json()
    assert "rsi" in blob.lower()
