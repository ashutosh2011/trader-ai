"""Canonical TradingView-compatible reference formulas for test fixtures.

Sources:
- EMA/SMA/MACD: pandas ``ewm(span, adjust=False)`` / ``rolling().mean()``
- RSI/ATR: Wilder RMA (``ta.rma``): SMA seed at bar ``period-1``, then alpha=1/period
- VWAP: cumulative (H+L+C)/3 * volume / cumulative volume
- BBands: SMA middle, sample stdev (ddof=1) bands
- Supertrend: TradingView Pine supertrend with ATR(period) Wilder smoothing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parent


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder RMA matching ``ta.rma`` (SMA seed, then exponential smoothing)."""
    length = len(series)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if length < period:
        return out
    alpha = 1.0 / period
    out.iloc[period - 1] = float(series.iloc[:period].mean())
    prev = out.iloc[period - 1]
    for i in range(period, length):
        prev = alpha * float(series.iloc[i]) + (1.0 - alpha) * float(prev)
        out.iloc[i] = prev
    return out


def reference_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def reference_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def reference_atr(candles: pd.DataFrame, period: int) -> pd.Series:
    high = candles["high"]
    low = candles["low"]
    close = candles["close"]
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    true_range.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def reference_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    return rsi.where(~both_zero, 50.0)


def reference_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        }
    )


def reference_vwap(candles: pd.DataFrame) -> pd.Series:
    typical = (candles["high"] + candles["low"] + candles["close"]) / 3.0
    pv = typical * candles["volume"]
    return pv.cumsum() / candles["volume"].cumsum()


def reference_bbands(
    close: pd.Series,
    period: int = 20,
    mult: float = 2.0,
) -> pd.DataFrame:
    middle = close.rolling(window=period, min_periods=period).mean()
    stdev = close.rolling(window=period, min_periods=period).std(ddof=1)
    return pd.DataFrame(
        {
            "upper": middle + mult * stdev,
            "middle": middle,
            "lower": middle - mult * stdev,
        }
    )


def reference_supertrend(
    candles: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    high = candles["high"].to_numpy(dtype=float)
    low = candles["low"].to_numpy(dtype=float)
    close = candles["close"].to_numpy(dtype=float)
    atr_vals = reference_atr(candles, period).to_numpy(dtype=float)
    n = len(close)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr_vals
    basic_lower = hl2 - multiplier * atr_vals

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)
    supertrend = np.full(n, np.nan)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = final_lower[0]

    for i in range(1, n):
        prev_upper = final_upper[i - 1]
        prev_lower = final_lower[i - 1]
        if close[i - 1] < prev_upper:
            final_upper[i] = min(basic_upper[i], prev_upper)
        else:
            final_upper[i] = basic_upper[i]
        if close[i - 1] > prev_lower:
            final_lower[i] = max(basic_lower[i], prev_lower)
        else:
            final_lower[i] = basic_lower[i]

        prev_dir = direction[i - 1]
        if prev_dir == -1 and close[i] > prev_upper:
            direction[i] = 1
        elif prev_dir == 1 and close[i] < prev_lower:
            direction[i] = -1
        else:
            direction[i] = prev_dir

        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.DataFrame(
        {
            "supertrend": supertrend,
            "direction": direction,
            "upper": final_upper,
            "lower": final_lower,
        },
        index=candles.index,
    )


def load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURE_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data
