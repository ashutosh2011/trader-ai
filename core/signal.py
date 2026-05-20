from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator

IST = ZoneInfo("Asia/Kolkata")


class Signal(BaseModel):
    """Trading signal emitted by a strategy."""

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
