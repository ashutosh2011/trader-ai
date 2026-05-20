"""Paper broker with simulated fills and slippage."""

from typing import Literal
from zoneinfo import ZoneInfo

import structlog
from pydantic import BaseModel, ConfigDict

from config.settings import AppSettings, PaperConfig
from core.signal import Signal
from execution.broker import (
    Broker,
    OrderResult,
    Position,
    deterministic_client_order_id,
)

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

PositionSide = Literal["LONG", "SHORT"]


class PaperAccount(BaseModel):
    """Paper trading account state."""

    model_config = ConfigDict(frozen=False)

    equity: float
    realized_pnl: float = 0.0


class PaperBroker(Broker):
    """Simulated broker: next-tick fill with configurable slippage bps.

    TRADEOFF: Fills at signal entry with slippage; bracket SL/target tracked
    logically on the position (same conservative same-bar rules as backtest
    are applied by the orchestrator when checking bar highs/lows).
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        paper_config: PaperConfig | None = None,
    ) -> None:
        app = settings or AppSettings()
        self._config = paper_config or app.paper
        self._positions: dict[str, Position] = {}
        self._account = PaperAccount(equity=self._config.account_equity)
        self._orders: list[OrderResult] = []

    @property
    def account(self) -> PaperAccount:
        return self._account

    @property
    def orders(self) -> list[OrderResult]:
        return list(self._orders)

    def place_bracket_order(self, signal: Signal, qty: int) -> OrderResult:
        """Fill entry with slippage; store SL/target on position."""
        if qty < 1:
            return OrderResult(
                client_order_id=deterministic_client_order_id(
                    signal.strategy_id, signal.ts, signal.symbol
                ),
                status="REJECTED",
                symbol=signal.symbol,
                side=signal.side,
                qty=0,
                fill_price=0.0,
                message="qty must be >= 1",
            )
        if signal.symbol in self._positions:
            return OrderResult(
                client_order_id=deterministic_client_order_id(
                    signal.strategy_id, signal.ts, signal.symbol
                ),
                status="REJECTED",
                symbol=signal.symbol,
                side=signal.side,
                qty=qty,
                fill_price=0.0,
                message="position already open",
            )

        fill = self._apply_slippage(signal.entry, signal.side)
        side: PositionSide = "LONG" if signal.side == "BUY" else "SHORT"
        position = Position(
            symbol=signal.symbol,
            side=side,
            qty=qty,
            entry_price=fill,
            stop_loss=signal.stop_loss,
            target=signal.target,
            strategy_id=signal.strategy_id,
            opened_at=signal.ts.astimezone(IST),
        )
        self._positions[signal.symbol] = position
        order_id = deterministic_client_order_id(signal.strategy_id, signal.ts, signal.symbol)
        result = OrderResult(
            client_order_id=order_id,
            status="FILLED",
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            fill_price=fill,
            message="paper_fill",
        )
        self._orders.append(result)
        logger.info(
            "paper_order_filled",
            order_id=order_id,
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            fill=fill,
        )
        return result

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def flatten_all(self) -> None:
        self._positions.clear()

    def close_position(self, symbol: str, exit_price: float, reason: str) -> float:
        """Close a position and update realized PnL; returns trade PnL."""
        position = self._positions.pop(symbol, None)
        if position is None:
            return 0.0
        if position.side == "LONG":
            pnl = (exit_price - position.entry_price) * position.qty
        else:
            pnl = (position.entry_price - exit_price) * position.qty
        self._account.realized_pnl += pnl
        self._account.equity += pnl
        logger.info("paper_position_closed", symbol=symbol, pnl=pnl, reason=reason)
        return pnl

    def check_bar_exits(self, symbol: str, high: float, low: float) -> tuple[float | None, str]:
        """Apply conservative SL-before-target bar exit rules.

        Returns:
            Tuple of (exit_price, reason) if exited, else (None, "").
        """
        position = self._positions.get(symbol)
        if position is None:
            return None, ""

        if position.side == "LONG":
            if low <= position.stop_loss:
                return position.stop_loss, "stop_loss"
            if high >= position.target:
                return position.target, "target"
        else:
            if high >= position.stop_loss:
                return position.stop_loss, "stop_loss"
            if low <= position.target:
                return position.target, "target"
        return None, ""

    def _apply_slippage(self, price: float, side: Literal["BUY", "SELL"]) -> float:
        bps = self._config.slippage_bps / 10_000.0
        if side == "BUY":
            return price * (1.0 + bps)
        return price * (1.0 - bps)
