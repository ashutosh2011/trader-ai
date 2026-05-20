"""EMA crossover example strategy with ATR-based stop and target."""

from datetime import datetime

from core.context import Context
from core.signal import Signal
from indicators.builtin.atr import ATR
from indicators.builtin.ema import EMA
from strategies.base import Strategy
from strategies.registry import register_strategy


@register_strategy
class EmaCrossover(Strategy):
    """EMA crossover strategy with ATR stop/target.

    Long when the fast EMA crosses above the slow EMA; short on the opposite
    cross. Stop loss is placed at entry ± 1×ATR; target at entry ± 2×ATR.

    Defaults: fast_period=12, slow_period=26, atr_period=14.
    """

    id = "ema_crossover"
    timeframe = "1m"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        atr_period: int = 14,
        symbol: str = "SYNTH",
    ) -> None:
        super().__init__()
        if fast_period >= slow_period:
            msg = "fast_period must be less than slow_period"
            raise ValueError(msg)
        self._symbol = symbol
        self._fast = EMA(fast_period)
        self._slow = EMA(slow_period)
        self._atr = ATR(atr_period)
        self.required_indicators = [self._fast, self._slow, self._atr]

    def on_bar(self, ctx: Context) -> list[Signal]:
        # Need previous bar for cross detection; all indicators must be warmed.
        min_bars = self.warmup_bars() + 1
        if ctx.bar_index < min_bars:
            return []

        indicators = self.precompute_indicators(ctx)
        fast = indicators[self._fast.param_key()]
        slow = indicators[self._slow.param_key()]
        atr = indicators[self._atr.param_key()]

        idx = ctx.bar_index
        fast_prev = float(fast.iloc[idx - 1])
        fast_curr = float(fast.iloc[idx])
        slow_prev = float(slow.iloc[idx - 1])
        slow_curr = float(slow.iloc[idx])
        atr_curr = float(atr.iloc[idx])
        entry = float(ctx.closed_bars["close"].iloc[idx])

        bullish = fast_prev <= slow_prev and fast_curr > slow_curr
        bearish = fast_prev >= slow_prev and fast_curr < slow_curr

        if not bullish and not bearish:
            return []

        ts = _ensure_ts(ctx.timestamp)
        snapshot = {
            "ema_fast": fast_curr,
            "ema_slow": slow_curr,
            "atr": atr_curr,
        }

        if bullish:
            return [
                Signal(
                    symbol=self._symbol,
                    side="BUY",
                    entry=entry,
                    stop_loss=entry - atr_curr,
                    target=entry + 2.0 * atr_curr,
                    timeframe=self.timeframe,
                    strategy_id=self.id,
                    reasons=["Fast EMA crossed above slow EMA"],
                    indicator_snapshot=snapshot,
                    confidence=_crossover_confidence(fast_curr, slow_curr, atr_curr),
                    ts=ts,
                )
            ]

        return [
            Signal(
                symbol=self._symbol,
                side="SELL",
                entry=entry,
                stop_loss=entry + atr_curr,
                target=entry - 2.0 * atr_curr,
                timeframe=self.timeframe,
                strategy_id=self.id,
                reasons=["Fast EMA crossed below slow EMA"],
                indicator_snapshot=snapshot,
                confidence=_crossover_confidence(fast_curr, slow_curr, atr_curr),
                ts=ts,
            )
        ]


EmaCrossoverStrategy = EmaCrossover


def _ensure_ts(value: datetime) -> datetime:
    if value.tzinfo is None:
        msg = "context timestamp must be timezone-aware"
        raise ValueError(msg)
    return value


def _crossover_confidence(fast: float, slow: float, atr: float) -> float:
    if atr <= 0:
        return 0.5
    separation = abs(fast - slow) / atr
    return float(min(1.0, max(0.0, separation / 2.0)))
