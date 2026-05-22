"""Tests for LLMTuner parsing and fallbacks."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from analyst.providers.mock import MockLLMProvider
from tuner.llm_tuner import DEFAULT_TUNING_PLAN, LLMTuner
from tuner.performance import StrategySymbolPerformance
from tuner.prompt import TuningContext

IST = ZoneInfo("Asia/Kolkata")


def _perf() -> list[StrategySymbolPerformance]:
    return [
        StrategySymbolPerformance(
            symbol="INFY",
            strategy_id="rsi_mean_revert",
            current_params={"oversold": 30},
            trades=(),
        )
    ]


def _ctx() -> TuningContext:
    return TuningContext(as_of=datetime.now(tz=IST), notes="")


@pytest.mark.asyncio
async def test_parse_valid_json() -> None:
    payload = {
        "name": "Tune",
        "summary_rationale": "Adjust RSI",
        "recommendations": [
            {
                "symbol": "INFY",
                "current_strategy_id": "rsi_mean_revert",
                "action": "modify_params",
                "params": {"oversold": 25},
                "rationale": "losses",
                "confidence": 0.8,
            }
        ],
    }
    provider = MockLLMProvider(json.dumps(payload), name="mock")
    plan, meta = await LLMTuner(provider).generate(_perf(), _ctx())
    assert meta.status == "ok"
    assert len(plan.recommendations) == 1
    assert plan.recommendations[0].params["oversold"] == 25


@pytest.mark.asyncio
async def test_parse_error_fallback() -> None:
    provider = MockLLMProvider("not json at all", name="mock")
    plan, meta = await LLMTuner(provider).generate(_perf(), _ctx())
    assert meta.status == "fallback_parse_error"
    assert plan == DEFAULT_TUNING_PLAN


@pytest.mark.asyncio
async def test_transport_fallback() -> None:
    class BrokenProvider(MockLLMProvider):
        async def complete(self, prompt: str) -> str:
            raise httpx.ConnectError("down")

    plan, meta = await LLMTuner(BrokenProvider("{}", name="broken")).generate(
        _perf(),
        _ctx(),
    )
    assert meta.status == "fallback_transport"
    assert plan == DEFAULT_TUNING_PLAN
