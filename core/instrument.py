"""NSE equity and F&O instrument metadata."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

InstrumentType = Literal["equity", "future", "option"]
OptionType = Literal["CE", "PE"]


class Instrument(BaseModel):
    """Tradable instrument metadata for NSE cash and derivatives."""

    symbol: str = Field(min_length=1, description="Trading symbol")
    exchange: str = Field(min_length=1, default="NSE")
    segment: Literal["EQ", "FO"] = "EQ"
    instrument_type: InstrumentType = "equity"
    tick_size: float = Field(gt=0, default=0.05)
    lot_size: int = Field(ge=1, default=1)
    expiry: date | None = None
    strike: float | None = None
    option_type: OptionType | None = None
    underlying: str | None = None

    @model_validator(mode="after")
    def validate_nse_fields(self) -> "Instrument":
        """Enforce NSE equity vs F&O field consistency."""
        if self.instrument_type == "equity":
            if self.segment != "EQ":
                msg = "equity instruments must use segment EQ"
                raise ValueError(msg)
            if self.expiry is not None or self.strike is not None or self.option_type is not None:
                msg = "equity instruments must not set expiry/strike/option_type"
                raise ValueError(msg)
            return self

        if self.segment != "FO":
            msg = "derivatives must use segment FO"
            raise ValueError(msg)
        if self.expiry is None:
            msg = "F&O instruments require expiry"
            raise ValueError(msg)
        if self.instrument_type == "option":
            if self.strike is None or self.option_type is None:
                msg = "options require strike and option_type"
                raise ValueError(msg)
        elif self.instrument_type == "future" and self.option_type is not None:
            msg = "futures must not set option_type"
            raise ValueError(msg)
        return self

    @classmethod
    def nse_equity(cls, symbol: str, *, lot_size: int = 1, tick_size: float = 0.05) -> "Instrument":
        """Build an NSE cash equity instrument."""
        return cls(
            symbol=symbol,
            exchange="NSE",
            segment="EQ",
            instrument_type="equity",
            lot_size=lot_size,
            tick_size=tick_size,
        )

    @classmethod
    def nse_future(
        cls,
        symbol: str,
        expiry: date,
        *,
        underlying: str,
        lot_size: int,
        tick_size: float = 0.05,
    ) -> "Instrument":
        """Build an NSE futures contract instrument."""
        return cls(
            symbol=symbol,
            exchange="NSE",
            segment="FO",
            instrument_type="future",
            expiry=expiry,
            underlying=underlying,
            lot_size=lot_size,
            tick_size=tick_size,
        )

    @classmethod
    def nse_option(
        cls,
        symbol: str,
        expiry: date,
        strike: float,
        option_type: OptionType,
        *,
        underlying: str,
        lot_size: int,
        tick_size: float = 0.05,
    ) -> "Instrument":
        """Build an NSE options contract instrument."""
        return cls(
            symbol=symbol,
            exchange="NSE",
            segment="FO",
            instrument_type="option",
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            underlying=underlying,
            lot_size=lot_size,
            tick_size=tick_size,
        )

    def round_qty(self, qty: int) -> int:
        """Round quantity to exchange lot size (minimum one lot)."""
        if qty < self.lot_size:
            return self.lot_size
        remainder = qty % self.lot_size
        if remainder == 0:
            return qty
        return qty + (self.lot_size - remainder)
