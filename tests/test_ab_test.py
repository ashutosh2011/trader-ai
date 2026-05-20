import pytest

from analyst.providers.mock import MockLLMProvider
from data.replay_feed import ReplayFeed
from orchestrator.ab_test import run_ab_test
from strategies.examples.ema_crossover import EmaCrossover
from tests.fixtures.bars import make_synthetic_bars


@pytest.mark.asyncio
async def test_ab_test_runs_both_paths() -> None:
    frame = make_synthetic_bars(300, seed=7)
    feed = ReplayFeed(frame)
    strategy = EmaCrossover(symbol="SYNTH")
    provider = MockLLMProvider(
        '{"action": "APPROVE", "size_multiplier": 0.5, "confidence": 0.7, "rationale": "half"}'
    )
    result = await run_ab_test(strategy, feed, analyst_provider=provider)
    assert result.rules_only.label == "rules_only"
    assert result.co_decide.label == "co_decide"
    assert result.rules_only.signals_seen >= 0
    assert result.co_decide.signals_seen >= 0


@pytest.mark.asyncio
async def test_ab_test_veto_reduces_orders() -> None:
    frame = make_synthetic_bars(400, seed=99)
    feed = ReplayFeed(frame)
    strategy = EmaCrossover(symbol="SYNTH")
    provider = MockLLMProvider(
        '{"action": "VETO", "size_multiplier": 0.0, "confidence": 1.0, "rationale": "veto all"}'
    )
    result = await run_ab_test(strategy, feed, analyst_provider=provider)
    assert result.co_decide.analyst_vetoed >= 0
    assert result.co_decide.orders_placed <= result.rules_only.orders_placed
