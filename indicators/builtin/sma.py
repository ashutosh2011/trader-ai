import pandas as pd

from indicators.base import Indicator
from indicators.registry import register_indicator


@register_indicator
class SMA(Indicator):
    """Simple moving average on close (rolling mean)."""

    name = "sma"

    def __init__(self, period: int) -> None:
        if period < 1:
            msg = "period must be >= 1"
            raise ValueError(msg)
        self.params: dict[str, int | float | str] = {"period": period}

    def compute(self, candles: pd.DataFrame) -> pd.Series:
        period = int(self.params["period"])
        return candles["close"].rolling(window=period, min_periods=period).mean()

    def warmup(self) -> int:
        return int(self.params["period"]) - 1
