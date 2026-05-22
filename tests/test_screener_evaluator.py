"""Evaluator tests on handcrafted DataFrames."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from screener.evaluator import evaluate
from screener.schema import (
    CompareTo,
    IndicatorFilter,
    PriceChangeFilter,
    ScreenerFilter,
    ScreenerFormula,
    VolumeFilter,
)

IST = ZoneInfo("Asia/Kolkata")


def _ohlcv(count: int, *, close_pattern: np.ndarray) -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2024-01-01 09:15:00",
        periods=count,
        freq="1min",
        tz=IST,
    )
    close = close_pattern
    open_ = np.empty(count)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(count, 1_000.0)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _formula(filters: list[ScreenerFilter], *, side: str = "long") -> ScreenerFormula:
    return ScreenerFormula(
        name="t",
        timeframe="day",
        side_bias=side,  # type: ignore[arg-type]
        rationale="t",
        filters=filters,
    )


def test_rsi_below_30_only_picks_oversold_symbol() -> None:
    # A: monotonically declining → very low RSI.
    a_close = np.linspace(150.0, 50.0, 100)
    # B: monotonically rising → very high RSI.
    b_close = np.linspace(50.0, 150.0, 100)
    candles = {
        "OVERSOLD": _ohlcv(100, close_pattern=a_close),
        "OVERBOUGHT": _ohlcv(100, close_pattern=b_close),
    }
    formula = _formula(
        [
            IndicatorFilter(
                indicator="rsi", params={"period": 14}, op="<", value=30.0
            )
        ]
    )
    results = evaluate(formula, candles)
    symbols = [r.symbol for r in results]
    assert symbols == ["OVERSOLD"]
    assert results[0].matches[0].value < 30.0
    assert results[0].matches[0].threshold == 30.0


def test_compare_to_close_vs_sma() -> None:
    # Symbol RISING: close finishes well above 50-SMA.
    rising = np.linspace(100.0, 200.0, 100)
    # Symbol FALLING: close finishes well below 50-SMA.
    falling = np.linspace(200.0, 100.0, 100)
    candles = {
        "RISING": _ohlcv(100, close_pattern=rising),
        "FALLING": _ohlcv(100, close_pattern=falling),
    }
    formula = _formula(
        [
            IndicatorFilter(
                indicator="close",
                op=">",
                compare_to=CompareTo(indicator="sma", params={"period": 50}),
            )
        ]
    )
    results = evaluate(formula, candles)
    symbols = [r.symbol for r in results]
    assert symbols == ["RISING"]


def test_volume_filter_x_avg_uses_rolling_mean() -> None:
    close = np.linspace(100.0, 110.0, 100)
    frame = _ohlcv(100, close_pattern=close)
    frame.loc[frame.index[-1], "volume"] = 5_000.0  # 5x baseline avg
    formula = _formula(
        [VolumeFilter(op=">", value_x_avg=2.0, avg_window=20)]
    )
    results = evaluate(formula, {"BURST": frame})
    assert len(results) == 1
    assert results[0].symbol == "BURST"
    assert results[0].matches[0].value == pytest.approx(5_000.0)
    assert isinstance(results[0].matches[0].threshold, str)
    assert "avg(20)" in results[0].matches[0].threshold


def test_volume_filter_x_avg_skips_when_rolling_mean_zero() -> None:
    close = np.linspace(100.0, 110.0, 100)
    frame = _ohlcv(100, close_pattern=close)
    frame.loc[:, "volume"] = 0.0  # rolling mean is zero
    formula = _formula(
        [VolumeFilter(op=">", value_x_avg=1.5, avg_window=20)]
    )
    results = evaluate(formula, {"DEAD": frame})
    assert results == []


def test_price_change_filter_lookback() -> None:
    # Build a close series where the last bar is 5% above the bar 5 ago.
    close = np.full(100, 100.0)
    close[-1] = 105.0
    frame = _ohlcv(100, close_pattern=close)
    formula = _formula(
        [PriceChangeFilter(window=5, op=">", value_pct=3.0)]
    )
    results = evaluate(formula, {"PUMP": frame})
    assert len(results) == 1
    assert results[0].matches[0].value == pytest.approx(5.0)


def test_missing_ohlcv_column_skips_symbol_silently() -> None:
    close = np.linspace(100.0, 110.0, 100)
    frame = _ohlcv(100, close_pattern=close).drop(columns=["volume"])
    formula = _formula(
        [IndicatorFilter(indicator="rsi", params={"period": 14}, op="<", value=99.0)]
    )
    results = evaluate(formula, {"INCOMPLETE": frame})
    assert results == []


def test_too_few_bars_skips_symbol_silently() -> None:
    close = np.linspace(100.0, 110.0, 30)
    frame = _ohlcv(30, close_pattern=close)
    formula = _formula(
        [IndicatorFilter(indicator="rsi", params={"period": 14}, op="<", value=99.0)]
    )
    results = evaluate(formula, {"SHORT": frame}, min_bars=60)
    assert results == []


def test_unknown_indicator_skips_symbol_silently() -> None:
    close = np.linspace(100.0, 110.0, 100)
    frame = _ohlcv(100, close_pattern=close)
    formula = _formula(
        [
            IndicatorFilter(
                indicator="moonbeam",
                op=">",
                value=1.0,
            )
        ]
    )
    results = evaluate(formula, {"X": frame})
    assert results == []


def test_all_filters_must_pass() -> None:
    rising = np.linspace(100.0, 200.0, 100)
    frame = _ohlcv(100, close_pattern=rising)
    formula = _formula(
        [
            IndicatorFilter(
                indicator="close",
                op=">",
                compare_to=CompareTo(indicator="sma", params={"period": 50}),
            ),
            # Impossible threshold — should kill the pick.
            IndicatorFilter(
                indicator="rsi",
                params={"period": 14},
                op="<",
                value=1.0,
            ),
        ]
    )
    results = evaluate(formula, {"R": frame})
    assert results == []


def test_results_carry_bars_and_timestamp() -> None:
    close = np.linspace(100.0, 200.0, 100)
    frame = _ohlcv(100, close_pattern=close)
    formula = _formula(
        [
            IndicatorFilter(
                indicator="close",
                op=">",
                compare_to=CompareTo(indicator="sma", params={"period": 50}),
            )
        ]
    )
    results = evaluate(formula, {"R": frame})
    assert len(results) == 1
    assert results[0].bars_evaluated == 100
    assert results[0].last_bar_ts.tzinfo is not None


def test_multiple_filters_capture_all_matches() -> None:
    rising = np.linspace(100.0, 200.0, 100)
    frame = _ohlcv(100, close_pattern=rising)
    # Volume burst on the last bar.
    frame.loc[frame.index[-1], "volume"] = 5_000.0
    formula = _formula(
        [
            IndicatorFilter(
                indicator="close",
                op=">",
                compare_to=CompareTo(indicator="sma", params={"period": 50}),
            ),
            VolumeFilter(op=">", value_x_avg=2.0, avg_window=20),
        ]
    )
    results = evaluate(formula, {"R": frame})
    assert len(results) == 1
    assert len(results[0].matches) == 2
    assert results[0].matches[0].filter_index == 0
    assert results[0].matches[1].filter_index == 1
