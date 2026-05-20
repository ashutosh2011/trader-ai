from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class Context:
    """
    Read-only strategy context for a single bar.

    Strategies and indicators must use only closed bars (index <= bar_index).
    """

    symbol: str
    bars: pd.DataFrame
    bar_index: int
    timestamp: datetime
    timeframe: str

    @property
    def closed_bars(self) -> pd.DataFrame:
        """Bars up to and including the current bar (no lookahead)."""
        return self.bars.iloc[: self.bar_index + 1].copy()

    @property
    def current_bar(self) -> pd.Series:
        return self.bars.iloc[self.bar_index]
