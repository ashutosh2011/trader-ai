"""MACD trend-following example strategy with ATR-based stop and target."""

import math
from datetime import datetime
from typing import cast

import pandas as pd

from core.context import Context
from core.signal import Signal
from indicators.builtin.atr import ATR
from indicators.builtin.macd import MACD
from strategies.base import Strategy
from strategies.registry import register_strategy


@register_strategy
class MacdTrend(Strategy):
    """MACD signal-line cross with histogram filter and ATR stop/target.

    Long when the MACD line crosses above the signal line and the current
    histogram is positive; short on the opposite cross with a negative
    histogram. Stop loss is placed at entry ± 1×ATR; target at entry ± 2×ATR.

    Defaults: macd_fast=12, macd_slow=26, macd_signal=9, atr_period=14,
    stop_atr_mult=1.0, target_atr_mult=2.0.
    """

    id = "macd_trend"
    timeframe = "1m"

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        stop_atr_mult: float = 1.0,
        target_atr_mult: float = 2.0,
        symbol: str = "SYNTH",
    ) -> None:
        super().__init__()
        if macd_fast < 1 or macd_slow < 1 or macd_signal < 1 or atr_period < 1:
            msg = "macd_fast, macd_slow, macd_signal, and atr_period must be >= 1"
            raise ValueError(msg)
        if macd_fast >= macd_slow:
            msg = "macd_fast must be less than macd_slow"
            raise ValueError(msg)
        if stop_atr_mult <= 0 or target_atr_mult <= 0:
            msg = "stop_atr_mult and target_atr_mult must be > 0"
            raise ValueError(msg)
        self._symbol = symbol
        self._stop_mult = stop_atr_mult
        self._target_mult = target_atr_mult
        self._macd = MACD(fast=macd_fast, slow=macd_slow, signal=macd_signal)
        self._atr = ATR(atr_period)
        self.required_indicators = [self._macd, self._atr]

    def on_bar(self, ctx: Context) -> list[Signal]:
        # Need previous bar for cross detection; all indicators must be warmed.
        min_bars = self.warmup_bars() + 1
        if ctx.bar_index < min_bars:
            return []

        indicators = self.precompute_indicators(ctx)
        macd_frame = cast(pd.DataFrame, indicators[self._macd.param_key()])
        atr = indicators[self._atr.param_key()]

        idx = ctx.bar_index
        macd_prev = float(macd_frame["macd"].iloc[idx - 1])
        macd_curr = float(macd_frame["macd"].iloc[idx])
        signal_prev = float(macd_frame["signal"].iloc[idx - 1])
        signal_curr = float(macd_frame["signal"].iloc[idx])
        hist_curr = float(macd_frame["histogram"].iloc[idx])
        atr_curr = float(atr.iloc[idx])
        entry = float(ctx.closed_bars["close"].iloc[idx])

        if not math.isfinite(atr_curr) or atr_curr <= 0:
            return []

        cross_up = macd_prev <= signal_prev and macd_curr > signal_curr
        cross_down = macd_prev >= signal_prev and macd_curr < signal_curr
        bullish = cross_up and hist_curr > 0
        bearish = cross_down and hist_curr < 0

        if not bullish and not bearish:
            return []

        ts = _ensure_ts(ctx.timestamp)
        snapshot = {
            "macd": macd_curr,
            "signal": signal_curr,
            "histogram": hist_curr,
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
                    reasons=["MACD crossed above signal with positive histogram"],
                    indicator_snapshot=snapshot,
                    confidence=_macd_confidence(hist_curr, atr_curr),
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
                reasons=["MACD crossed below signal with negative histogram"],
                indicator_snapshot=snapshot,
                confidence=_macd_confidence(hist_curr, atr_curr),
                ts=ts,
            )
        ]


MacdTrendStrategy = MacdTrend


def _ensure_ts(value: datetime) -> datetime:
    if value.tzinfo is None:
        msg = "context timestamp must be timezone-aware"
        raise ValueError(msg)
    return value


def _macd_confidence(histogram: float, atr: float) -> float:
    if atr <= 0:
        return 0.5
    separation = abs(histogram) / atr
    return float(min(1.0, max(0.0, separation)))
