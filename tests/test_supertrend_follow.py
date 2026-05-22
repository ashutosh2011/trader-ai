import pandas as pd
import pytest

import strategies  # noqa: F401 — register examples
from core.context import Context
from core.signal import Signal
from strategies.examples.supertrend_follow import SupertrendFollow


def test_supertrend_follow_invalid_periods() -> None:
    with pytest.raises(ValueError, match="st_period and atr_period must be >= 1"):
        SupertrendFollow(st_period=0)
    with pytest.raises(ValueError, match="st_period and atr_period must be >= 1"):
        SupertrendFollow(atr_period=0)


def test_supertrend_follow_invalid_multiplier() -> None:
    with pytest.raises(ValueError, match="st_multiplier must be > 0"):
        SupertrendFollow(st_multiplier=0.0)
    with pytest.raises(ValueError, match="st_multiplier must be > 0"):
        SupertrendFollow(st_multiplier=-2.0)


def test_supertrend_follow_invalid_atr_mults() -> None:
    with pytest.raises(ValueError, match="stop_atr_mult and target_atr_mult must be > 0"):
        SupertrendFollow(stop_atr_mult=0.0)
    with pytest.raises(ValueError, match="stop_atr_mult and target_atr_mult must be > 0"):
        SupertrendFollow(target_atr_mult=-1.0)


def test_supertrend_follow_warmup_positive() -> None:
    strategy = SupertrendFollow(st_period=10, atr_period=14)
    assert strategy.warmup_bars() > 0


def test_supertrend_follow_emits_signal_on_flip() -> None:
    """Decline drives direction to -1, then a sharp rally flips it to +1."""
    frame = _decline_then_rip_bars()
    strategy = SupertrendFollow(
        st_period=5,
        st_multiplier=2.0,
        atr_period=5,
        stop_atr_mult=1.0,
        target_atr_mult=2.0,
        symbol="X",
    )
    signals: list[Signal] = []
    for bar_index in range(len(frame)):
        signals.extend(strategy.on_bar(_make_context(frame, bar_index)))
    assert signals, "expected at least one Supertrend flip signal"
    first = signals[0]
    assert first.side == "BUY"
    assert first.stop_loss < first.entry < first.target
    expected_stop = first.entry - 1.0 * first.indicator_snapshot["atr"]
    expected_target = first.entry + 2.0 * first.indicator_snapshot["atr"]
    assert first.stop_loss == pytest.approx(expected_stop)
    assert first.target == pytest.approx(expected_target)
    assert first.indicator_snapshot["direction"] == pytest.approx(1.0)
    assert "supertrend" in first.indicator_snapshot
    assert first.confidence == pytest.approx(0.6)
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
        periods=40,
        freq="1min",
        tz="Asia/Kolkata",
    )
    declining = [100.0 - 1.5 * i for i in range(20)]
    rising = [declining[-1] + 2.5 * (i + 1) for i in range(20)]
    close = declining + rising
    open_ = [close[0], *close[:-1]]
    high = [c + 0.4 for c in close]
    low = [c - 0.4 for c in close]
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
