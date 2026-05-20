import pandas as pd

from indicators.base import Indicator
from indicators.registry import register_indicator


@register_indicator
class EMA(Indicator):
    """Exponential moving average on close (adjust=False, span-based)."""

    name = "ema"

    def __init__(self, span: int) -> None:
        if span < 1:
            msg = "span must be >= 1"
            raise ValueError(msg)
        self.params: dict[str, int | float | str] = {"span": span}

    def compute(self, candles: pd.DataFrame) -> pd.Series:
        return candles["close"].ewm(span=int(self.params["span"]), adjust=False).mean()

    def warmup(self) -> int:
        return int(self.params["span"]) - 1
