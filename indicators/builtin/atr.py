import pandas as pd

from indicators.base import Indicator
from indicators.registry import register_indicator


@register_indicator
class ATR(Indicator):
    """Average True Range using Wilder smoothing (adjust=False)."""

    name = "atr"

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            msg = "period must be >= 1"
            raise ValueError(msg)
        self.params: dict[str, int | float | str] = {"period": period}

    def compute(self, candles: pd.DataFrame) -> pd.Series:
        period = int(self.params["period"])
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

    def warmup(self) -> int:
        return int(self.params["period"])
