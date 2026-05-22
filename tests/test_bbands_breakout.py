import pandas as pd
import pytest

import strategies  # noqa: F401 — register examples
from core.context import Context
from core.signal import Signal
from strategies.examples.bbands_breakout import BBandsBreakout


def test_bbands_breakout_invalid_periods() -> None:
    with pytest.raises(ValueError, match="bb_period and atr_period must be >= 1"):
        BBandsBreakout(bb_period=0)
    with pytest.raises(ValueError, match="bb_period and atr_period must be >= 1"):
        BBandsBreakout(atr_period=0)


def test_bbands_breakout_invalid_mult() -> None:
    with pytest.raises(ValueError, match="bb_mult must be > 0"):
        BBandsBreakout(bb_mult=0.0)
    with pytest.raises(ValueError, match="bb_mult must be > 0"):
        BBandsBreakout(bb_mult=-1.5)


def test_bbands_breakout_invalid_atr_mults() -> None:
    with pytest.raises(ValueError, match="stop_atr_mult and target_atr_mult must be > 0"):
        BBandsBreakout(stop_atr_mult=0.0)
    with pytest.raises(ValueError, match="stop_atr_mult and target_atr_mult must be > 0"):
        BBandsBreakout(target_atr_mult=-1.0)


def test_bbands_breakout_warmup_positive() -> None:
    strategy = BBandsBreakout(bb_period=20, atr_period=14)
    assert strategy.warmup_bars() > 0


def test_bbands_breakout_emits_long_on_upper_break() -> None:
    """Stable prices form tight bands, then a single spike breaks above upper."""
    frame = _quiet_then_spike_bars()
    strategy = BBandsBreakout(
        bb_period=10,
        bb_mult=2.0,
        atr_period=5,
        stop_atr_mult=1.0,
        target_atr_mult=2.0,
        symbol="X",
    )
    signals: list[Signal] = []
    for bar_index in range(len(frame)):
        signals.extend(strategy.on_bar(_make_context(frame, bar_index)))
    assert signals, "expected at least one breakout signal"
    first = signals[0]
    assert first.side == "BUY"
    assert first.stop_loss < first.entry < first.target
    expected_stop = first.entry - 1.0 * first.indicator_snapshot["atr"]
    expected_target = first.entry + 2.0 * first.indicator_snapshot["atr"]
    assert first.stop_loss == pytest.approx(expected_stop)
    assert first.target == pytest.approx(expected_target)
    for key in ("bb_upper", "bb_middle", "bb_lower", "atr"):
        assert key in first.indicator_snapshot
    assert first.reasons


def _make_context(frame: pd.DataFrame, bar_index: int) -> Context:
    return Context(
        symbol=str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "X",
        bars=frame,
        bar_index=bar_index,
        timestamp=frame["timestamp"].iloc[bar_index].to_pydatetime(),
        timeframe="1m",
    )


def _quiet_then_spike_bars() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01 09:15:00",
        periods=40,
        freq="1min",
        tz="Asia/Kolkata",
    )
    # Alternating tiny moves keep the standard deviation strictly positive while
    # the mean stays near 100. Then a sharp jump must breach the upper band.
    quiet = [100.0 + (0.05 if i % 2 == 0 else -0.05) for i in range(30)]
    spike = [110.0, 112.0, 114.0, 116.0, 118.0, 120.0, 122.0, 124.0, 126.0, 128.0]
    close = quiet + spike
    open_ = [close[0], *close[:-1]]
    high = [c + 0.1 for c in close]
    low = [c - 0.1 for c in close]
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
