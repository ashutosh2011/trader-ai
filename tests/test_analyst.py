import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from analyst.analyst import ANALYST_TIMEOUT_SEC, Analyst, _parse_verdict
from analyst.provider import LLMProvider
from analyst.providers.mock import MockLLMProvider
from core.context import Context
from core.signal import Signal
from tests.fixtures.bars import make_synthetic_bars

IST = ZoneInfo("Asia/Kolkata")


def _signal() -> Signal:
    return Signal(
        symbol="SYNTH",
        side="BUY",
        entry=100.0,
        stop_loss=99.0,
        target=102.0,
        timeframe="1m",
        strategy_id="test",
        reasons=["cross"],
        indicator_snapshot={"ema_fast": 101.0},
        confidence=0.7,
        ts=datetime(2024, 1, 1, 10, 0, tzinfo=IST),
    )


def _ctx() -> Context:
    frame = make_synthetic_bars(50)
    return Context(
        symbol="SYNTH",
        bars=frame,
        bar_index=49,
        timestamp=frame["timestamp"].iloc[49].to_pydatetime(),
        timeframe="1m",
    )


def test_parse_veto_and_shrink() -> None:
    veto = _parse_verdict(
        '{"action": "VETO", "size_multiplier": 1.0, "confidence": 0.9, "rationale": "no"}',
        provider="mock",
        latency_ms=1,
    )
    assert veto.action == "VETO"
    shrink = _parse_verdict(
        '{"action": "SHRINK", "size_multiplier": 1.5, "confidence": 0.5, "rationale": "caution"}',
        provider="mock",
        latency_ms=1,
    )
    assert shrink.action == "SHRINK"
    assert shrink.size_multiplier == 1.0


@pytest.mark.asyncio
async def test_timeout_fallback() -> None:
    class SlowProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "slow"

        async def complete(self, prompt: str) -> str:
            await asyncio.sleep(ANALYST_TIMEOUT_SEC + 0.5)
            return "{}"

    analyst = Analyst(SlowProvider())
    verdict = await analyst.analyze(_signal(), _ctx())
    assert verdict.provider == "fallback"
    assert verdict.action == "APPROVE"
    assert verdict.size_multiplier == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_mock_approve() -> None:
    provider = MockLLMProvider(
        '{"action": "APPROVE", "size_multiplier": 0.6, "confidence": 0.8, "rationale": "ok"}'
    )
    verdict = await Analyst(provider).analyze(_signal(), _ctx())
    assert verdict.action == "APPROVE"
    assert verdict.size_multiplier == 0.6
    assert verdict.provider == "mock"


@pytest.mark.asyncio
async def test_parse_error_becomes_veto() -> None:
    provider = MockLLMProvider("not json at all")
    verdict = await Analyst(provider).analyze(_signal(), _ctx())
    assert verdict.action == "VETO"
    assert verdict.provider == "fallback_parse_error"
    assert verdict.size_multiplier == 0.0


@pytest.mark.asyncio
async def test_transport_error_falls_back_to_approve() -> None:
    class HttpxBoom(LLMProvider):
        @property
        def name(self) -> str:
            return "httpx_boom"

        async def complete(self, prompt: str) -> str:
            raise httpx.ConnectError("network down")

    verdict = await Analyst(HttpxBoom()).analyze(_signal(), _ctx())
    assert verdict.provider == "fallback"
    assert verdict.action == "APPROVE"
    assert verdict.size_multiplier == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_unexpected_exception_becomes_veto() -> None:
    class BoomProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "broken"

        async def complete(self, prompt: str) -> str:
            raise RuntimeError("boom")

    verdict = await Analyst(BoomProvider()).analyze(_signal(), _ctx())
    assert verdict.action == "VETO"
    assert verdict.provider == "fallback_unexpected"
    assert verdict.size_multiplier == 0.0
