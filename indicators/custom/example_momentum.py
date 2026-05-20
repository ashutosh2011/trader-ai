"""Example custom indicator: price momentum (close - close.shift(n)).

Copy this file to add proprietary logic later. Register with ``@register_indicator``
and import the module (or ``indicators.custom``) so the registry picks it up.
"""

import pandas as pd

from indicators.base import Indicator
from indicators.registry import register_indicator


@register_indicator
class PriceMomentum(Indicator):
    """Simple derived indicator: close minus close shifted by ``period`` bars."""

    name = "price_momentum"

    def __init__(self, period: int = 10) -> None:
        if period < 1:
            msg = "period must be >= 1"
            raise ValueError(msg)
        self.params: dict[str, int | float | str] = {"period": period}

    def compute(self, candles: pd.DataFrame) -> pd.Series:
        period = int(self.params["period"])
        close = candles["close"]
        return close - close.shift(period)

    def warmup(self) -> int:
        return int(self.params["period"])
