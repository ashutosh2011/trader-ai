"""Tests for the layered analyst JSON extractor and parse fallbacks."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from analyst.analyst import Analyst, _extract_json_object, _parse_verdict
from analyst.providers.anthropic import parse_verdict_json
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


def test_parse_verdict_json_codeblock_legacy() -> None:
    raw = (
        '```json\n{"action": "SHRINK", "size_multiplier": 0.4, '
        '"confidence": 0.6, "rationale": "x"}\n```'
    )
    verdict = parse_verdict_json(raw, provider="anthropic", latency_ms=10)
    assert verdict.action == "SHRINK"
    assert verdict.size_multiplier == 0.4
    assert verdict.provider == "anthropic"


def test_extract_plain_object() -> None:
    raw = '{"action": "APPROVE", "size_multiplier": 0.8}'
    assert _extract_json_object(raw) == raw


def test_extract_fenced_object() -> None:
    raw = '```\n{"action": "APPROVE", "size_multiplier": 0.8}\n```'
    assert '"action"' in _extract_json_object(raw)


def test_extract_fenced_json_language_tag() -> None:
    raw = '```json\n{"action": "APPROVE", "size_multiplier": 0.5}\n```'
    extracted = _extract_json_object(raw)
    assert '"action"' in extracted
    # No leading 'json' tag should remain.
    assert "json" not in extracted.split("{", 1)[0]


def test_extract_leading_commentary() -> None:
    raw = (
        "I will analyze this trade carefully. "
        'Here is my verdict: {"action": "APPROVE", "size_multiplier": 0.5}'
    )
    extracted = _extract_json_object(raw)
    assert extracted.startswith("{")
    assert extracted.endswith("}")


def test_extract_trailing_commentary() -> None:
    raw = (
        '{"action": "VETO", "size_multiplier": 0.0, "rationale": "no"} '
        "(this is my final answer)"
    )
    extracted = _extract_json_object(raw)
    assert extracted.startswith("{")
    assert extracted.endswith("}")
    assert "(this" not in extracted


def test_extract_nested_object() -> None:
    raw = (
        'prefix {"action": "APPROVE", "size_multiplier": 0.5, '
        '"indicators": {"ema": 1.0}} suffix'
    )
    extracted = _extract_json_object(raw)
    assert '"indicators"' in extracted
    assert '"ema"' in extracted


def test_extract_garbage_raises() -> None:
    with pytest.raises(ValueError):
        _extract_json_object("this is not json at all")


def test_parse_missing_size_multiplier_defaults_to_07() -> None:
    raw = '{"action": "APPROVE", "confidence": 0.8, "rationale": "ok"}'
    verdict = _parse_verdict(raw, provider="mock", latency_ms=1)
    assert verdict.size_multiplier == pytest.approx(0.7)


def test_parse_invalid_action_raises() -> None:
    raw = '{"action": "MAYBE", "size_multiplier": 0.5}'
    with pytest.raises(ValueError, match="invalid action"):
        _parse_verdict(raw, provider="mock", latency_ms=1)


@pytest.mark.asyncio
async def test_analyst_garbage_input_produces_veto() -> None:
    provider = MockLLMProvider("complete garbage with no json")
    verdict = await Analyst(provider).analyze(_signal(), _ctx())
    assert verdict.action == "VETO"
    assert verdict.provider == "fallback_parse_error"


@pytest.mark.asyncio
async def test_analyst_commentary_then_json_parses() -> None:
    raw = (
        "Looking at the trade setup... "
        '{"action": "APPROVE", "size_multiplier": 0.8, "confidence": 0.7, "rationale": "ok"}'
    )
    verdict = await Analyst(MockLLMProvider(raw)).analyze(_signal(), _ctx())
    assert verdict.action == "APPROVE"
    assert verdict.size_multiplier == pytest.approx(0.8)
