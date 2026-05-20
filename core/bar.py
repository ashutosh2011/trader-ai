from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

IST = ZoneInfo("Asia/Kolkata")


class Bar(BaseModel):
    """Single OHLCV candle with timezone-aware timestamp."""

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "timestamp must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(IST)

    @model_validator(mode="after")
    def high_gte_low(self) -> "Bar":
        if self.high < self.low:
            msg = "high must be >= low"
            raise ValueError(msg)
        return self
