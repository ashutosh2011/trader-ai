import pandas as pd
import pytest

import strategies  # noqa: F401 — register examples
from core.context import Context
from core.signal import Signal
from strategies.examples.rsi_mean_revert import RsiMeanRevert


def test_rsi_mean_revert_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="0 < oversold < overbought < 100"):
        RsiMeanRevert(oversold=70.0, overbought=30.0)
    with pytest.raises(ValueError, match="0 < oversold < overbought < 100"):
        RsiMeanRevert(oversold=-1.0, overbought=70.0)
    with pytest.raises(ValueError, match="0 < oversold < overbought < 100"):
        RsiMeanRevert(oversold=30.0, overbought=110.0)


def test_rsi_mean_revert_invalid_periods() -> None:
    with pytest.raises(ValueError, match="rsi_period and atr_period must be >= 1"):
        RsiMeanRevert(rsi_period=0)
    with pytest.raises(ValueError, match="rsi_period and atr_period must be >= 1"):
        RsiMeanRevert(atr_period=0)


def test_rsi_mean_revert_invalid_mults() -> None:
    with pytest.raises(ValueError, match="stop_atr_mult and target_atr_mult must be > 0"):
        RsiMeanRevert(stop_atr_mult=0.0)
    with pytest.raises(ValueError, match="stop_atr_mult and target_atr_mult must be > 0"):
        RsiMeanRevert(target_atr_mult=-1.0)


def test_rsi_mean_revert_warmup_positive() -> None:
    strategy = RsiMeanRevert(rsi_period=14, atr_period=14)
    assert strategy.warmup_bars() > 0


def test_rsi_mean_revert_emits_long_on_cross_up() -> None:
    """Sharp drop then steady recovery forces an RSI cross up through 30."""
    frame = _drop_then_recover_bars()
    strategy = RsiMeanRevert(
        rsi_period=5,
        atr_period=5,
        oversold=30.0,
        overbought=70.0,
        stop_atr_mult=1.0,
        target_atr_mult=1.5,
        symbol="X",
    )
    signals: list[Signal] = []
    for bar_index in range(len(frame)):
        signals.extend(strategy.on_bar(_make_context(frame, bar_index)))
    assert signals, "expected at least one long signal on RSI cross up"
    first = signals[0]
    assert first.side == "BUY"
    assert first.stop_loss < first.entry < first.target
    expected_stop = first.entry - 1.0 * first.indicator_snapshot["atr"]
    expected_target = first.entry + 1.5 * first.indicator_snapshot["atr"]
    assert first.stop_loss == pytest.approx(expected_stop)
    assert first.target == pytest.approx(expected_target)
    assert "rsi" in first.indicator_snapshot
    assert "rsi_prev" in first.indicator_snapshot
    assert first.reasons


def _make_context(frame: pd.DataFrame, bar_index: int) -> Context:
    return Context(
        symbol=str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "X",
        bars=frame,
        bar_index=bar_index,
        timestamp=frame["timestamp"].iloc[bar_index].to_pydatetime(),
        timeframe="1m",
    )


def _drop_then_recover_bars() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01 09:15:00",
        periods=40,
        freq="1min",
        tz="Asia/Kolkata",
    )
    # 20 sharp declining bars push RSI toward 0, then 20 rising bars drive
    # a crossing back above oversold (30).
    declining = [100.0 - 2.0 * i for i in range(20)]
    rising = [declining[-1] + 2.0 * (i + 1) for i in range(20)]
    close = declining + rising
    open_ = [close[0], *close[:-1]]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
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
