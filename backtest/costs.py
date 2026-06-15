"""Transaction-cost model applied by the backtest engine.

Real fills are never free: brokers charge commission and the market moves
against you between the signal and the fill (slippage). Modelling both turns
a gross-PnL backtest into a net-PnL one, which is the only honest basis for
comparing strategies. Both inputs are expressed as a percentage of traded
notional so they scale with price and quantity without needing per-symbol
tick/lot metadata.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Side = Literal["LONG", "SHORT"]


class CostModel(BaseModel):
    """Commission + slippage charged per fill, as a fraction of notional.

    TRADEOFF: A flat percentage is a deliberate simplification over a
    full Indian-market cost stack (brokerage + STT + exchange txn +
    GST + SEBI + stamp duty). For relative strategy comparison the
    blended percentage is what matters, and it keeps the form to two
    intuitive numbers. ``commission_pct`` is charged on **each** side
    (entry and exit); ``slippage_pct`` worsens **each** fill price.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    commission_pct: float = Field(default=0.0, ge=0.0, le=10.0)
    slippage_pct: float = Field(default=0.0, ge=0.0, le=10.0)

    @property
    def is_zero(self) -> bool:
        """True when the model imposes no cost (the cheap fast path)."""
        return self.commission_pct == 0.0 and self.slippage_pct == 0.0

    def entry_fill_price(self, side: Side, price: float) -> float:
        """Return the slippage-adjusted entry price for ``side``.

        Slippage always works against the trader: a long pays up, a
        short sells lower than the quoted level.
        """
        if self.slippage_pct == 0.0:
            return price
        factor = self.slippage_pct / 100.0
        return price * (1.0 + factor) if side == "LONG" else price * (1.0 - factor)

    def exit_fill_price(self, side: Side, price: float) -> float:
        """Return the slippage-adjusted exit price for ``side``.

        Closing a long sells lower, closing a short buys back higher.
        """
        if self.slippage_pct == 0.0:
            return price
        factor = self.slippage_pct / 100.0
        return price * (1.0 - factor) if side == "LONG" else price * (1.0 + factor)

    def commission(self, entry_price: float, exit_price: float, qty: int) -> float:
        """Total commission for the round trip (both legs)."""
        if self.commission_pct == 0.0:
            return 0.0
        rate = self.commission_pct / 100.0
        return rate * (abs(entry_price) + abs(exit_price)) * qty


ZERO_COST = CostModel()


__all__ = ["ZERO_COST", "CostModel", "Side"]
