import pandas as pd

from indicators.base import Indicator
from indicators.registry import register_indicator


@register_indicator
class VWAP(Indicator):
    """Cumulative volume-weighted average price (no session reset).

    # TRADEOFF: TradingView session VWAP resets each session; this implementation
    # is cumulative over the entire ``candles`` DataFrame passed to ``compute``.
    """

    name = "vwap"

    def __init__(self) -> None:
        self.params: dict[str, int | float | str] = {}

    def compute(self, candles: pd.DataFrame) -> pd.Series:
        typical = (candles["high"] + candles["low"] + candles["close"]) / 3.0
        pv = typical * candles["volume"]
        cum_pv = pv.cumsum()
        cum_vol = candles["volume"].cumsum()
        return cum_pv / cum_vol

    def warmup(self) -> int:
        return 0
