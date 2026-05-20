import numpy as np
import pandas as pd

from indicators.base import Indicator
from indicators.builtin.atr import ATR
from indicators.registry import register_indicator


@register_indicator
class Supertrend(Indicator):
    """ATR-based Supertrend (TradingView-style band ratchet and trend flip)."""

    name = "supertrend"

    def __init__(self, period: int = 10, multiplier: float = 3.0) -> None:
        if period < 1:
            msg = "period must be >= 1"
            raise ValueError(msg)
        if multiplier <= 0:
            msg = "multiplier must be > 0"
            raise ValueError(msg)
        self.params: dict[str, int | float | str] = {
            "period": period,
            "multiplier": multiplier,
        }

    def compute(self, candles: pd.DataFrame) -> pd.DataFrame:
        period = int(self.params["period"])
        multiplier = float(self.params["multiplier"])
        high = candles["high"].to_numpy(dtype=float)
        low = candles["low"].to_numpy(dtype=float)
        close = candles["close"].to_numpy(dtype=float)
        atr_vals = ATR(period=period).compute(candles).to_numpy(dtype=float)
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

    def warmup(self) -> int:
        return int(self.params["period"])
