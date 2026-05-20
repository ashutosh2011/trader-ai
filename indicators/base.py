from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd


class Indicator(ABC):
    """Technical indicator with vectorized compute and warmup metadata."""

    name: ClassVar[str]
    params: dict[str, int | float | str]

    @abstractmethod
    def compute(self, candles: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Compute indicator values from OHLCV candles."""

    @abstractmethod
    def warmup(self) -> int:
        """Minimum closed bars required before values are meaningful."""

    def param_key(self) -> str:
        param_parts = [f"{key}={value}" for key, value in sorted(self.params.items())]
        return f"{self.name}({','.join(param_parts)})"
