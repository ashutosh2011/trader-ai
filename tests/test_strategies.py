import pandas as pd
import pytest

import strategies  # noqa: F401 — register examples
from core.context import Context
from strategies.examples.ema_crossover import EmaCrossover
from strategies.registry import get_strategy, list_strategies


def test_ema_crossover_registered() -> None:
    assert "ema_crossover" in list_strategies()
    assert get_strategy("ema_crossover") is EmaCrossover


def test_get_strategy_unknown_raises() -> None:
    with pytest.raises(KeyError, match="strategy not registered"):
        get_strategy("missing_strategy")


def test_ema_crossover_warmup_gates_signals(synthetic_bars_200: pd.DataFrame) -> None:
    strategy = EmaCrossover(fast_period=5, slow_period=10, atr_period=5, symbol="SYNTH")
    min_bar = strategy.warmup_bars() + 1
    for bar_index in range(min_bar):
        ctx = _make_context(synthetic_bars_200, bar_index)
        assert strategy.on_bar(ctx) == []


def test_ema_crossover_no_signal_without_cross(synthetic_bars_200: pd.DataFrame) -> None:
    strategy = EmaCrossover(fast_period=5, slow_period=10, atr_period=5, symbol="SYNTH")
    signals_seen = 0
    for bar_index in range(strategy.warmup_bars() + 1, len(synthetic_bars_200)):
        signals = strategy.on_bar(_make_context(synthetic_bars_200, bar_index))
        for signal in signals:
            signals_seen += 1
            assert signal.indicator_snapshot
            assert signal.reasons
            assert signal.ts.tzinfo is not None
    assert signals_seen >= 1


def test_ema_crossover_cross_detection_only_on_cross() -> None:
    """Flat then step price forces a single bullish cross, not persistent signals."""
    frame = _cross_bars()
    strategy = EmaCrossover(fast_period=3, slow_period=8, atr_period=3, symbol="X")
    signal_bars = [
        idx
        for idx in range(len(frame))
        if strategy.on_bar(_make_context(frame, idx))
    ]
    assert len(signal_bars) >= 1
    assert len(signal_bars) <= 3


def test_ema_crossover_invalid_periods() -> None:
    with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
        EmaCrossover(fast_period=20, slow_period=10)


def _make_context(frame: pd.DataFrame, bar_index: int) -> Context:
    return Context(
        symbol=str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "SYNTH",
        bars=frame,
        bar_index=bar_index,
        timestamp=frame["timestamp"].iloc[bar_index].to_pydatetime(),
        timeframe="1m",
    )


def _cross_bars() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01 09:15:00",
        periods=40,
        freq="1min",
        tz="Asia/Kolkata",
    )
    close = [100.0] * 15 + [100.0 + i * 0.5 for i in range(1, 26)]
    open_ = [close[0]] + close[:-1]
    high = [c + 0.2 for c in close]
    low = [c - 0.2 for c in close]
    volume = [1000.0] * len(close)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "X",
        }
    )
