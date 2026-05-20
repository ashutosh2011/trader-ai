"""Strategy base class and indicator precomputation helpers."""

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from core.context import Context
from core.signal import Signal
from indicators.base import Indicator


def compute_indicators(
    ctx: Context,
    indicators: list[Indicator],
    *,
    cache: dict[str, pd.Series | pd.DataFrame] | None = None,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Compute indicator values from closed bars only (no lookahead).

    Args:
        ctx: Current bar context; only bars up to ``bar_index`` are used.
        indicators: Indicators to compute.
        cache: Optional cache keyed by ``Indicator.param_key()``. Values are
            recomputed when the cached series is shorter than closed bars.

    Returns:
        Mapping of indicator param keys to computed series or frames.
    """
    closed = ctx.closed_bars
    result: dict[str, pd.Series | pd.DataFrame] = {}
    for indicator in indicators:
        key = indicator.param_key()
        if cache is not None and key in cache and len(cache[key]) >= len(closed):
            result[key] = cache[key]
            continue
        computed = indicator.compute(closed)
        result[key] = computed
        if cache is not None:
            cache[key] = computed
    return result


class Strategy(ABC):
    """Pure strategy logic driven by bar context (no I/O, no wall-clock time).

    Subclasses implement ``on_bar`` with synchronous logic only. Use
    ``compute_indicators`` or ``precompute_indicators`` to derive indicator
    values from ``ctx.closed_bars`` without lookahead.
    """

    id: ClassVar[str]
    timeframe: ClassVar[str]
    required_indicators: list[Indicator]

    def __init__(self) -> None:
        self._indicator_cache: dict[str, pd.Series | pd.DataFrame] = {}

    def precompute_indicators(self, ctx: Context) -> dict[str, pd.Series | pd.DataFrame]:
        """Return indicator values for ``required_indicators``, using an instance cache.

        Args:
            ctx: Current bar context.

        Returns:
            Mapping of indicator param keys to computed series or frames.
        """
        return compute_indicators(ctx, self.required_indicators, cache=self._indicator_cache)

    def warmup_bars(self) -> int:
        """Minimum ``bar_index`` (inclusive) before indicators are fully warmed."""
        if not self.required_indicators:
            return 0
        return max(indicator.warmup() for indicator in self.required_indicators)

    @abstractmethod
    def on_bar(self, ctx: Context) -> list[Signal]:
        """Evaluate the current bar and return zero or more signals.

        Args:
            ctx: Read-only context for the bar being evaluated.

        Returns:
            Signals emitted on this bar (may be empty).
        """
