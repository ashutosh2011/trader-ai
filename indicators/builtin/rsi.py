import numpy as np
import pandas as pd

from indicators.base import Indicator
from indicators.registry import register_indicator


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder RMA (TradingView ``ta.rma``): SMA seed, then ``alpha = 1/period``."""
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


@register_indicator
class RSI(Indicator):
    """Relative Strength Index using Wilder smoothing (TradingView standard)."""

    name = "rsi"

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            msg = "period must be >= 1"
            raise ValueError(msg)
        self.params: dict[str, int | float | str] = {"period": period}

    def compute(self, candles: pd.DataFrame) -> pd.Series:
        period = int(self.params["period"])
        delta = candles["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = _wilder_rma(gain, period)
        avg_loss = _wilder_rma(loss, period)
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(avg_loss != 0, 100.0)
        rsi = rsi.where(avg_gain != 0, 0.0)
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        rsi = rsi.where(~both_zero, 50.0)
        return rsi

    def warmup(self) -> int:
        return int(self.params["period"]) - 1
