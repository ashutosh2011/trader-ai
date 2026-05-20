import pandas as pd

from indicators.base import Indicator
from indicators.registry import register_indicator


@register_indicator
class MACD(Indicator):
    """MACD line, signal line, and histogram (EMA-based, default 12/26/9)."""

    name = "macd"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> None:
        if fast < 1 or slow < 1 or signal < 1:
            msg = "fast, slow, and signal must be >= 1"
            raise ValueError(msg)
        if fast >= slow:
            msg = "fast span must be less than slow span"
            raise ValueError(msg)
        self.params: dict[str, int | float | str] = {
            "fast": fast,
            "slow": slow,
            "signal": signal,
        }

    def compute(self, candles: pd.DataFrame) -> pd.DataFrame:
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])
        signal_span = int(self.params["signal"])
        close = candles["close"]
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {
                "macd": macd_line,
                "signal": signal_line,
                "histogram": histogram,
            },
            index=candles.index,
        )

    def warmup(self) -> int:
        slow = int(self.params["slow"])
        signal_span = int(self.params["signal"])
        return slow - 1 + signal_span - 1
