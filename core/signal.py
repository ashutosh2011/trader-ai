"""Trading signal model with direction-aware price validation."""

import math
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

IST = ZoneInfo("Asia/Kolkata")


class Signal(BaseModel):
    """Trading signal emitted by a strategy.

    Direction validation enforces, for a BUY: ``stop_loss < entry < target``;
    for a SELL: ``target < entry < stop_loss``. All three prices must be
    strictly positive.
    """

    symbol: str
    side: Literal["BUY", "SELL"]
    entry: float
    stop_loss: float
    target: float
    qty: int | None = None
    timeframe: str
    strategy_id: str
    reasons: list[str]
    indicator_snapshot: dict[str, float]
    confidence: float = Field(ge=0, le=1)
    ts: datetime

    @field_validator("ts")
    @classmethod
    def ts_must_be_tz_aware_ist(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "ts must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(IST)

    @model_validator(mode="after")
    def validate_direction(self) -> "Signal":
        """Enforce direction-consistent entry/stop/target with positive prices."""
        if self.entry <= 0 or self.stop_loss <= 0 or self.target <= 0:
            msg = (
                f"entry/stop_loss/target must be positive (got entry={self.entry}, "
                f"stop_loss={self.stop_loss}, target={self.target})"
            )
            raise ValueError(msg)
        if self.side == "BUY":
            if not (self.stop_loss < self.entry < self.target):
                msg = (
                    f"BUY signal requires stop_loss < entry < target "
                    f"(got stop_loss={self.stop_loss}, entry={self.entry}, "
                    f"target={self.target})"
                )
                raise ValueError(msg)
        elif not (self.target < self.entry < self.stop_loss):
            msg = (
                f"SELL signal requires target < entry < stop_loss "
                f"(got target={self.target}, entry={self.entry}, "
                f"stop_loss={self.stop_loss})"
            )
            raise ValueError(msg)
        return self

    def with_tick_rounding(self, tick_size: float) -> "Signal":
        """Return a copy with prices snapped to the exchange tick grid.

        Rounding rules:
            * ``entry``: nearest tick using banker's rounding (Python's
              built-in :func:`round`, round-half-to-even).
            * BUY ``stop_loss`` and ``target``: floor to tick.
            * SELL ``stop_loss`` and ``target``: ceil to tick.

        Args:
            tick_size: Exchange tick size; must be strictly positive.

        Returns:
            A new :class:`Signal` with rounded prices that still satisfies
            the direction validator.

        Raises:
            ValueError: If ``tick_size`` is non-positive, or if rounding
                collapses the prices into a state that violates the direction
                validator (e.g. tick larger than the entry/stop/target spread).
        """
        if tick_size <= 0:
            msg = f"tick_size must be positive (got {tick_size})"
            raise ValueError(msg)

        # TRADEOFF: stops floor for BUY / ceil for SELL and targets mirror that
        # direction. The rationale is "never make a stop tighter than the
        # original by rounding, and never make a target further away" — both
        # rules bias the rounded signal toward giving the trade more room to
        # work and slightly less reward, which is the conservative choice.
        entry = _round_to_tick(self.entry, tick_size)
        if self.side == "BUY":
            stop_loss = _floor_to_tick(self.stop_loss, tick_size)
            target = _floor_to_tick(self.target, tick_size)
        else:
            stop_loss = _ceil_to_tick(self.stop_loss, tick_size)
            target = _ceil_to_tick(self.target, tick_size)

        return Signal(
            symbol=self.symbol,
            side=self.side,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            qty=self.qty,
            timeframe=self.timeframe,
            strategy_id=self.strategy_id,
            reasons=list(self.reasons),
            indicator_snapshot=dict(self.indicator_snapshot),
            confidence=self.confidence,
            ts=self.ts,
        )


def _round_to_tick(price: float, tick: float) -> float:
    """Snap ``price`` to the nearest tick (banker's rounding on .5)."""
    return round(price / tick) * tick


def _floor_to_tick(price: float, tick: float) -> float:
    """Floor ``price`` down to the nearest tick."""
    return math.floor(price / tick) * tick


def _ceil_to_tick(price: float, tick: float) -> float:
    """Ceil ``price`` up to the nearest tick."""
    return math.ceil(price / tick) * tick
