"""Bollinger Bands breakout example strategy with ATR-based stop and target."""

import math
from datetime import datetime
from typing import cast

import pandas as pd

from core.context import Context
from core.signal import Signal
from indicators.builtin.atr import ATR
from indicators.builtin.bbands import BBands
from strategies.base import Strategy
from strategies.registry import register_strategy


@register_strategy
class BBandsBreakout(Strategy):
    """Bollinger Bands breakout strategy with ATR stop/target.

    Long when the close crosses above the upper band; short on the opposite
    cross below the lower band. Stop loss is placed at entry ± 1×ATR; target
    at entry ± 2×ATR.

    Defaults: bb_period=20, bb_mult=2.0, atr_period=14, stop_atr_mult=1.0,
    target_atr_mult=2.0.
    """

    id = "bbands_breakout"
    timeframe = "1m"

    def __init__(
        self,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        atr_period: int = 14,
        stop_atr_mult: float = 1.0,
        target_atr_mult: float = 2.0,
        symbol: str = "SYNTH",
    ) -> None:
        super().__init__()
        if bb_period < 1 or atr_period < 1:
            msg = "bb_period and atr_period must be >= 1"
            raise ValueError(msg)
        if bb_mult <= 0:
            msg = "bb_mult must be > 0"
            raise ValueError(msg)
        if stop_atr_mult <= 0 or target_atr_mult <= 0:
            msg = "stop_atr_mult and target_atr_mult must be > 0"
            raise ValueError(msg)
        self._symbol = symbol
        self._stop_mult = stop_atr_mult
        self._target_mult = target_atr_mult
        self._bbands = BBands(period=bb_period, mult=bb_mult)
        self._atr = ATR(atr_period)
        self.required_indicators = [self._bbands, self._atr]

    def on_bar(self, ctx: Context) -> list[Signal]:
        # Need previous bar for cross detection; all indicators must be warmed.
        min_bars = self.warmup_bars() + 1
        if ctx.bar_index < min_bars:
            return []

        indicators = self.precompute_indicators(ctx)
        bbands = cast(pd.DataFrame, indicators[self._bbands.param_key()])
        atr = indicators[self._atr.param_key()]
        closed_close = ctx.closed_bars["close"]

        idx = ctx.bar_index
        prev_close = float(closed_close.iloc[idx - 1])
        curr_close = float(closed_close.iloc[idx])
        prev_upper = float(bbands["upper"].iloc[idx - 1])
        prev_lower = float(bbands["lower"].iloc[idx - 1])
        curr_upper = float(bbands["upper"].iloc[idx])
        curr_middle = float(bbands["middle"].iloc[idx])
        curr_lower = float(bbands["lower"].iloc[idx])
        atr_curr = float(atr.iloc[idx])

        if not math.isfinite(atr_curr) or atr_curr <= 0:
            return []
        if not (math.isfinite(prev_upper) and math.isfinite(prev_lower)):
            return []

        bullish = prev_close <= prev_upper and curr_close > curr_upper
        bearish = prev_close >= prev_lower and curr_close < curr_lower

        if not bullish and not bearish:
            return []

        ts = _ensure_ts(ctx.timestamp)
        snapshot = {
            "bb_upper": curr_upper,
            "bb_middle": curr_middle,
            "bb_lower": curr_lower,
            "atr": atr_curr,
        }
        band_width = curr_upper - curr_lower

        if bullish:
            stop_loss = curr_close - self._stop_mult * atr_curr
            target = curr_close + self._target_mult * atr_curr
            if stop_loss >= curr_close or target <= curr_close:
                return []
            return [
                Signal(
                    symbol=self._symbol,
                    side="BUY",
                    entry=curr_close,
                    stop_loss=stop_loss,
                    target=target,
                    timeframe=self.timeframe,
                    strategy_id=self.id,
                    reasons=["Close broke above upper Bollinger band"],
                    indicator_snapshot=snapshot,
                    confidence=_bbands_confidence(
                        curr_close, curr_upper, band_width, atr_curr
                    ),
                    ts=ts,
                )
            ]

        stop_loss = curr_close + self._stop_mult * atr_curr
        target = curr_close - self._target_mult * atr_curr
        if stop_loss <= curr_close or target >= curr_close:
            return []
        return [
            Signal(
                symbol=self._symbol,
                side="SELL",
                entry=curr_close,
                stop_loss=stop_loss,
                target=target,
                timeframe=self.timeframe,
                strategy_id=self.id,
                reasons=["Close broke below lower Bollinger band"],
                indicator_snapshot=snapshot,
                confidence=_bbands_confidence(
                    curr_close, curr_lower, band_width, atr_curr
                ),
                ts=ts,
            )
        ]


BBandsBreakoutStrategy = BBandsBreakout


def _ensure_ts(value: datetime) -> datetime:
    if value.tzinfo is None:
        msg = "context timestamp must be timezone-aware"
        raise ValueError(msg)
    return value


def _bbands_confidence(
    close: float, band: float, band_width: float, atr: float
) -> float:
    # TRADEOFF: prefer band width as the denominator (it reflects current
    # volatility on the strategy's own indicator); fall back to ATR if the
    # band has collapsed to zero, mirroring the ``_crossover_confidence`` shape.
    denom = band_width if band_width > 0 else atr
    if denom <= 0:
        return 0.5
    separation = abs(close - band) / denom
    return float(min(1.0, max(0.0, separation)))
