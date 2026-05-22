import pandas as pd
import pytest

import strategies  # noqa: F401 — register examples
from core.context import Context
from core.signal import Signal
from strategies.examples.macd_trend import MacdTrend


def test_macd_trend_invalid_periods() -> None:
    with pytest.raises(
        ValueError,
        match="macd_fast, macd_slow, macd_signal, and atr_period must be >= 1",
    ):
        MacdTrend(macd_fast=0)
    with pytest.raises(
        ValueError,
        match="macd_fast, macd_slow, macd_signal, and atr_period must be >= 1",
    ):
        MacdTrend(atr_period=0)


def test_macd_trend_invalid_fast_slow_order() -> None:
    with pytest.raises(ValueError, match="macd_fast must be less than macd_slow"):
        MacdTrend(macd_fast=26, macd_slow=12)


def test_macd_trend_invalid_atr_mults() -> None:
    with pytest.raises(ValueError, match="stop_atr_mult and target_atr_mult must be > 0"):
        MacdTrend(stop_atr_mult=0.0)
    with pytest.raises(ValueError, match="stop_atr_mult and target_atr_mult must be > 0"):
        MacdTrend(target_atr_mult=-1.0)


def test_macd_trend_warmup_positive() -> None:
    strategy = MacdTrend(macd_fast=12, macd_slow=26, macd_signal=9, atr_period=14)
    assert strategy.warmup_bars() > 0


def test_macd_trend_emits_long_on_bullish_cross() -> None:
    """Decline then sharp recovery forces a MACD/signal cross with positive histogram."""
    frame = _decline_then_rip_bars()
    strategy = MacdTrend(
        macd_fast=3,
        macd_slow=8,
        macd_signal=3,
        atr_period=5,
        stop_atr_mult=1.0,
        target_atr_mult=2.0,
        symbol="X",
    )
    signals: list[Signal] = []
    for bar_index in range(len(frame)):
        signals.extend(strategy.on_bar(_make_context(frame, bar_index)))
    assert signals, "expected at least one MACD bullish cross signal"
    first = signals[0]
    assert first.side == "BUY"
    assert first.stop_loss < first.entry < first.target
    expected_stop = first.entry - 1.0 * first.indicator_snapshot["atr"]
    expected_target = first.entry + 2.0 * first.indicator_snapshot["atr"]
    assert first.stop_loss == pytest.approx(expected_stop)
    assert first.target == pytest.approx(expected_target)
    for key in ("macd", "signal", "histogram", "atr"):
        assert key in first.indicator_snapshot
    assert first.indicator_snapshot["histogram"] > 0
    assert first.reasons


def _make_context(frame: pd.DataFrame, bar_index: int) -> Context:
    return Context(
        symbol=str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "X",
        bars=frame,
        bar_index=bar_index,
        timestamp=frame["timestamp"].iloc[bar_index].to_pydatetime(),
        timeframe="1m",
    )


def _decline_then_rip_bars() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01 09:15:00",
        periods=50,
        freq="1min",
        tz="Asia/Kolkata",
    )
    declining = [100.0 - 1.0 * i for i in range(25)]
    rising = [declining[-1] + 2.0 * (i + 1) for i in range(25)]
    close = declining + rising
    open_ = [close[0], *close[:-1]]
    high = [c + 0.3 for c in close]
    low = [c - 0.3 for c in close]
    volume = [1_000.0] * len(close)
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
