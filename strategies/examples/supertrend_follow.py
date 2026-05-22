"""Supertrend direction-flip example strategy with ATR-based stop and target."""

import math
from datetime import datetime
from typing import cast

import pandas as pd

from core.context import Context
from core.signal import Signal
from indicators.builtin.atr import ATR
from indicators.builtin.supertrend import Supertrend
from strategies.base import Strategy
from strategies.registry import register_strategy


@register_strategy
class SupertrendFollow(Strategy):
    """Supertrend direction-flip follower with ATR stop/target.

    Long when Supertrend direction flips from -1 to +1; short on the opposite
    flip. Stop loss is placed at entry ± 1×ATR; target at entry ± 2×ATR. ATR
    is computed independently for sizing — Supertrend already embeds its own
    ATR for the band itself.

    Defaults: st_period=10, st_multiplier=3.0, atr_period=14,
    stop_atr_mult=1.0, target_atr_mult=2.0.
    """

    id = "supertrend_follow"
    timeframe = "1m"

    def __init__(
        self,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        atr_period: int = 14,
        stop_atr_mult: float = 1.0,
        target_atr_mult: float = 2.0,
        symbol: str = "SYNTH",
    ) -> None:
        super().__init__()
        if st_period < 1 or atr_period < 1:
            msg = "st_period and atr_period must be >= 1"
            raise ValueError(msg)
        if st_multiplier <= 0:
            msg = "st_multiplier must be > 0"
            raise ValueError(msg)
        if stop_atr_mult <= 0 or target_atr_mult <= 0:
            msg = "stop_atr_mult and target_atr_mult must be > 0"
            raise ValueError(msg)
        self._symbol = symbol
        self._stop_mult = stop_atr_mult
        self._target_mult = target_atr_mult
        self._supertrend = Supertrend(period=st_period, multiplier=st_multiplier)
        self._atr = ATR(atr_period)
        self.required_indicators = [self._supertrend, self._atr]

    def on_bar(self, ctx: Context) -> list[Signal]:
        # Need previous bar for flip detection; all indicators must be warmed.
        min_bars = self.warmup_bars() + 1
        if ctx.bar_index < min_bars:
            return []

        indicators = self.precompute_indicators(ctx)
        st_frame = cast(pd.DataFrame, indicators[self._supertrend.param_key()])
        atr = indicators[self._atr.param_key()]

        idx = ctx.bar_index
        prev_dir = int(st_frame["direction"].iloc[idx - 1])
        curr_dir = int(st_frame["direction"].iloc[idx])
        curr_st = float(st_frame["supertrend"].iloc[idx])
        atr_curr = float(atr.iloc[idx])
        entry = float(ctx.closed_bars["close"].iloc[idx])

        if not math.isfinite(atr_curr) or atr_curr <= 0:
            return []

        bullish = prev_dir == -1 and curr_dir == 1
        bearish = prev_dir == 1 and curr_dir == -1

        if not bullish and not bearish:
            return []

        ts = _ensure_ts(ctx.timestamp)
        snapshot = {
            "supertrend": curr_st,
            "direction": float(curr_dir),
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
                    reasons=["Supertrend flipped from -1 to +1"],
                    indicator_snapshot=snapshot,
                    confidence=_supertrend_flip_confidence(),
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
                reasons=["Supertrend flipped from +1 to -1"],
                indicator_snapshot=snapshot,
                confidence=_supertrend_flip_confidence(),
                ts=ts,
            )
        ]


SupertrendFollowStrategy = SupertrendFollow


def _ensure_ts(value: datetime) -> datetime:
    if value.tzinfo is None:
        msg = "context timestamp must be timezone-aware"
        raise ValueError(msg)
    return value


def _supertrend_flip_confidence() -> float:
    # TRADEOFF: Supertrend direction is a binary signal — there is no continuous
    # magnitude to scale into a 0..1 score, so we report a constant moderate
    # confidence instead of fabricating one from unrelated indicators.
    return 0.6
