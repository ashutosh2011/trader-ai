"""RSI mean-reversion example strategy with ATR-based stop and target."""

import math
from datetime import datetime

from core.context import Context
from core.signal import Signal
from indicators.builtin.atr import ATR
from indicators.builtin.rsi import RSI
from strategies.base import Strategy
from strategies.registry import register_strategy


@register_strategy
class RsiMeanRevert(Strategy):
    """RSI mean-reversion strategy with ATR stop/target.

    Long when RSI crosses up through the oversold threshold; short on the
    opposite cross through the overbought threshold. Stop loss is placed at
    entry ± ``stop_atr_mult``×ATR; target at entry ± ``target_atr_mult``×ATR.
    Defaults are tighter than trend-following templates because mean-reverting
    moves give back quickly.

    Defaults: rsi_period=14, oversold=30, overbought=70, atr_period=14,
    stop_atr_mult=1.0, target_atr_mult=1.5.
    """

    id = "rsi_mean_revert"
    timeframe = "1m"

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        atr_period: int = 14,
        stop_atr_mult: float = 1.0,
        target_atr_mult: float = 1.5,
        symbol: str = "SYNTH",
    ) -> None:
        super().__init__()
        if rsi_period < 1 or atr_period < 1:
            msg = "rsi_period and atr_period must be >= 1"
            raise ValueError(msg)
        if not 0.0 < oversold < overbought < 100.0:
            msg = "thresholds must satisfy 0 < oversold < overbought < 100"
            raise ValueError(msg)
        if stop_atr_mult <= 0 or target_atr_mult <= 0:
            msg = "stop_atr_mult and target_atr_mult must be > 0"
            raise ValueError(msg)
        self._symbol = symbol
        self._oversold = oversold
        self._overbought = overbought
        self._stop_mult = stop_atr_mult
        self._target_mult = target_atr_mult
        self._rsi = RSI(rsi_period)
        self._atr = ATR(atr_period)
        self.required_indicators = [self._rsi, self._atr]

    def on_bar(self, ctx: Context) -> list[Signal]:
        # Need previous bar for cross detection; all indicators must be warmed.
        min_bars = self.warmup_bars() + 1
        if ctx.bar_index < min_bars:
            return []

        indicators = self.precompute_indicators(ctx)
        rsi = indicators[self._rsi.param_key()]
        atr = indicators[self._atr.param_key()]

        idx = ctx.bar_index
        rsi_prev = float(rsi.iloc[idx - 1])
        rsi_curr = float(rsi.iloc[idx])
        atr_curr = float(atr.iloc[idx])
        entry = float(ctx.closed_bars["close"].iloc[idx])

        if not math.isfinite(atr_curr) or atr_curr <= 0:
            return []

        bullish = rsi_prev <= self._oversold and rsi_curr > self._oversold
        bearish = rsi_prev >= self._overbought and rsi_curr < self._overbought

        if not bullish and not bearish:
            return []

        ts = _ensure_ts(ctx.timestamp)
        snapshot = {
            "rsi": rsi_curr,
            "rsi_prev": rsi_prev,
            "atr": atr_curr,
        }

        if bullish:
            stop_loss = entry - self._stop_mult * atr_curr
            target = entry + self._target_mult * atr_curr
            if stop_loss >= entry or target <= entry:
                return []
            return [
                Signal(
                    symbol=self._symbol,
                    side="BUY",
                    entry=entry,
                    stop_loss=stop_loss,
                    target=target,
                    timeframe=self.timeframe,
                    strategy_id=self.id,
                    reasons=[
                        f"RSI crossed up through oversold ({self._oversold:g})"
                    ],
                    indicator_snapshot=snapshot,
                    confidence=_rsi_revert_confidence(rsi_curr, self._oversold, atr_curr),
                    ts=ts,
                )
            ]

        stop_loss = entry + self._stop_mult * atr_curr
        target = entry - self._target_mult * atr_curr
        if stop_loss <= entry or target >= entry:
            return []
        return [
            Signal(
                symbol=self._symbol,
                side="SELL",
                entry=entry,
                stop_loss=stop_loss,
                target=target,
                timeframe=self.timeframe,
                strategy_id=self.id,
                reasons=[
                    f"RSI crossed down through overbought ({self._overbought:g})"
                ],
                indicator_snapshot=snapshot,
                confidence=_rsi_revert_confidence(rsi_curr, self._overbought, atr_curr),
                ts=ts,
            )
        ]


RsiMeanRevertStrategy = RsiMeanRevert


def _ensure_ts(value: datetime) -> datetime:
    if value.tzinfo is None:
        msg = "context timestamp must be timezone-aware"
        raise ValueError(msg)
    return value


def _rsi_revert_confidence(rsi: float, threshold: float, atr: float) -> float:
    # TRADEOFF: RSI is unit-less (0..100) and ATR is in price units; we keep
    # the ``separation / atr`` shape of ``_crossover_confidence`` for symmetry
    # with other examples even though the two scales are not comparable.
    if atr <= 0:
        return 0.5
    separation = abs(rsi - threshold) / atr
    return float(min(1.0, max(0.0, separation / 2.0)))
