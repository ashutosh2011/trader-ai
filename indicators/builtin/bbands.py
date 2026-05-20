import pandas as pd

from indicators.base import Indicator
from indicators.registry import register_indicator


@register_indicator
class BBands(Indicator):
    """Bollinger Bands: upper, middle (SMA), lower (sample stdev, ``ddof=1``)."""

    name = "bbands"

    def __init__(self, period: int = 20, mult: float = 2.0) -> None:
        if period < 1:
            msg = "period must be >= 1"
            raise ValueError(msg)
        if mult <= 0:
            msg = "mult must be > 0"
            raise ValueError(msg)
        self.params: dict[str, int | float | str] = {"period": period, "mult": mult}

    def compute(self, candles: pd.DataFrame) -> pd.DataFrame:
        period = int(self.params["period"])
        mult = float(self.params["mult"])
        close = candles["close"]
        middle = close.rolling(window=period, min_periods=period).mean()
        stdev = close.rolling(window=period, min_periods=period).std(ddof=1)
        upper = middle + mult * stdev
        lower = middle - mult * stdev
        return pd.DataFrame(
            {"upper": upper, "middle": middle, "lower": lower},
            index=candles.index,
        )

    def warmup(self) -> int:
        return int(self.params["period"]) - 1
