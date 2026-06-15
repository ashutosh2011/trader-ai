"""Bar-walk backtest engine with next-bar-open fills."""

from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from pydantic import BaseModel, ConfigDict

from backtest.costs import ZERO_COST, CostModel
from core.context import Context
from core.signal import Signal
from strategies.base import Strategy

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

PositionSide = Literal["LONG", "SHORT"]


class ClosedTrade(BaseModel):
    """A completed round-trip trade."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    qty: int
    entry_bar: int
    exit_bar: int
    pnl: float
    exit_reason: str
    fees: float = 0.0


class EquityPoint(BaseModel):
    """Realized equity snapshot at a bar close."""

    model_config = ConfigDict(frozen=True)

    bar_index: int
    timestamp: datetime
    equity: float


class BacktestSummary(BaseModel):
    """Basic backtest statistics (see :mod:`backtest.metrics` for full analytics)."""

    model_config = ConfigDict(frozen=True)

    trade_count: int
    winning_trades: int
    losing_trades: int
    total_pnl: float


class BacktestResult(BaseModel):
    """Typed backtest output."""

    model_config = ConfigDict(frozen=True)

    closed_trades: list[ClosedTrade]
    equity_curve: list[EquityPoint]
    summary: BacktestSummary

    @property
    def trade_count(self) -> int:
        return self.summary.trade_count

    @property
    def total_pnl(self) -> float:
        return self.summary.total_pnl


class _OpenPosition:
    __slots__ = (
        "symbol",
        "side",
        "entry_price",
        "stop_loss",
        "target",
        "qty",
        "entry_bar",
    )

    def __init__(
        self,
        *,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        stop_loss: float,
        target: float,
        qty: int,
        entry_bar: int,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.target = target
        self.qty = qty
        self.entry_bar = entry_bar


class _PendingEntry:
    __slots__ = ("signal", "signal_bar")

    def __init__(self, signal: Signal, signal_bar: int) -> None:
        self.signal = signal
        self.signal_bar = signal_bar


class BacktestEngine:
    """Bar-walk backtester with next-bar-open fills and stop/target exits.

    TRADEOFF: Signals carry an ``entry`` price from the signal bar close, but
    fills occur at the **next bar open**. Reported trade entry prices reflect
    actual fill prices (open), not the signal's suggested entry.

    TRADEOFF: When stop loss and target are both reachable within the same bar,
    **stop loss is assumed to hit first** (conservative rule for both long and
    short positions).

    A :class:`~backtest.costs.CostModel` may be supplied to charge commission
    and apply slippage on every fill. The default is zero-cost, so existing
    callers see unchanged gross-PnL behaviour.
    """

    def __init__(self, qty: int = 1, *, cost_model: CostModel | None = None) -> None:
        if qty < 1:
            msg = "qty must be >= 1"
            raise ValueError(msg)
        self._qty = qty
        self._cost_model = cost_model or ZERO_COST

    def run(
        self,
        strategy: Strategy,
        bars: pd.DataFrame | Path | str,
    ) -> BacktestResult:
        """Run a backtest for ``strategy`` over ``bars``.

        Args:
            strategy: Strategy instance to evaluate each bar.
            bars: OHLCV DataFrame or path to a CSV with required columns.

        Returns:
            Backtest result with trades, equity curve, and summary stats.
        """
        frame = _normalize_bars(bars) if isinstance(bars, pd.DataFrame) else load_bars(bars)
        symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "UNKNOWN"
        closed: list[ClosedTrade] = []
        equity_curve: list[EquityPoint] = []
        realized_pnl = 0.0
        position: _OpenPosition | None = None
        pending: _PendingEntry | None = None

        for bar_index in range(len(frame)):
            row = frame.iloc[bar_index]
            ts = _parse_timestamp(row["timestamp"])

            if pending is not None and pending.signal_bar < bar_index:
                position, pending, reverse_trade = _fill_pending(
                    pending=pending,
                    position=position,
                    open_price=float(row["open"]),
                    bar_index=bar_index,
                    default_qty=self._qty,
                    cost_model=self._cost_model,
                )
                if reverse_trade is not None:
                    closed.append(reverse_trade)
                    realized_pnl += reverse_trade.pnl

            if position is not None:
                position, exit_trade = _check_exit(
                    position=position,
                    row=row,
                    bar_index=bar_index,
                    cost_model=self._cost_model,
                )
                if exit_trade is not None:
                    closed.append(exit_trade)
                    realized_pnl += exit_trade.pnl

            ctx = Context(
                symbol=symbol,
                bars=frame,
                bar_index=bar_index,
                timestamp=ts,
                timeframe=strategy.timeframe,
            )
            try:
                signals = strategy.on_bar(ctx)
            except Exception:
                logger.exception(
                    "strategy_on_bar_failed",
                    bar_index=bar_index,
                    strategy_id=strategy.id,
                )
                signals = []

            if signals and bar_index < len(frame) - 1:
                pending = _PendingEntry(signal=signals[0], signal_bar=bar_index)

            equity_curve.append(
                EquityPoint(bar_index=bar_index, timestamp=ts, equity=realized_pnl)
            )

        summary = _build_summary(closed)
        result = BacktestResult(
            closed_trades=closed,
            equity_curve=equity_curve,
            summary=summary,
        )
        logger.info(
            "backtest_complete",
            trades=result.trade_count,
            total_pnl=result.total_pnl,
        )
        return result


def load_bars(path: Path | str) -> pd.DataFrame:
    """Load OHLCV bars from CSV (timestamp, open, high, low, close, volume)."""
    return _normalize_bars(pd.read_csv(path))


def load_bars_csv(path: Path) -> pd.DataFrame:
    """Load OHLCV bars from CSV (alias for :func:`load_bars`)."""
    return load_bars(path)


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        msg = f"bars missing columns: {sorted(missing)}"
        raise ValueError(msg)
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(IST)
    return frame.reset_index(drop=True)


def _parse_timestamp(value: datetime | pd.Timestamp | str | int | float) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    py_ts = ts.to_pydatetime()
    if py_ts.tzinfo is None:
        msg = "timestamp must be timezone-aware"
        raise ValueError(msg)
    return py_ts.astimezone(IST)


def _fill_pending(
    *,
    pending: _PendingEntry,
    position: _OpenPosition | None,
    open_price: float,
    bar_index: int,
    default_qty: int,
    cost_model: CostModel = ZERO_COST,
) -> tuple[_OpenPosition | None, _PendingEntry | None, ClosedTrade | None]:
    signal = pending.signal
    reverse_trade: ClosedTrade | None = None
    if position is not None:
        reverse_trade = _close_position(
            position, open_price, bar_index, "signal_reverse", cost_model=cost_model
        )
        position = None

    side: PositionSide = "LONG" if signal.side == "BUY" else "SHORT"
    # Slippage is paid at the fill: the recorded entry price already
    # reflects the adverse move so downstream PnL stays consistent.
    entry_fill = cost_model.entry_fill_price(side, open_price)
    position = _OpenPosition(
        symbol=signal.symbol,
        side=side,
        entry_price=entry_fill,
        stop_loss=signal.stop_loss,
        target=signal.target,
        qty=signal.qty if signal.qty is not None else default_qty,
        entry_bar=bar_index,
    )
    return position, None, reverse_trade


def _check_exit(
    *,
    position: _OpenPosition,
    row: pd.Series,
    bar_index: int,
    cost_model: CostModel = ZERO_COST,
) -> tuple[_OpenPosition | None, ClosedTrade | None]:
    high = float(row["high"])
    low = float(row["low"])

    if position.side == "LONG":
        stop_hit = low <= position.stop_loss
        target_hit = high >= position.target
        if stop_hit:
            return None, _close_position(
                position, position.stop_loss, bar_index, "stop_loss", cost_model=cost_model
            )
        if target_hit:
            return None, _close_position(
                position, position.target, bar_index, "target", cost_model=cost_model
            )
        return position, None

    stop_hit = high >= position.stop_loss
    target_hit = low <= position.target
    if stop_hit:
        return None, _close_position(
            position, position.stop_loss, bar_index, "stop_loss", cost_model=cost_model
        )
    if target_hit:
        return None, _close_position(
            position, position.target, bar_index, "target", cost_model=cost_model
        )
    return position, None


def _close_position(
    position: _OpenPosition,
    exit_price: float,
    bar_index: int,
    reason: str,
    *,
    cost_model: CostModel = ZERO_COST,
) -> ClosedTrade:
    # The trigger level (stop/target/open) is reached; slippage then
    # worsens the actual exit fill before PnL and commission are booked.
    exit_fill = cost_model.exit_fill_price(position.side, exit_price)
    if position.side == "LONG":
        gross = (exit_fill - position.entry_price) * position.qty
    else:
        gross = (position.entry_price - exit_fill) * position.qty
    fees = cost_model.commission(position.entry_price, exit_fill, position.qty)
    return ClosedTrade(
        symbol=position.symbol,
        side=position.side,
        entry_price=position.entry_price,
        exit_price=exit_fill,
        qty=position.qty,
        entry_bar=position.entry_bar,
        exit_bar=bar_index,
        pnl=gross - fees,
        exit_reason=reason,
        fees=fees,
    )


def _build_summary(closed: list[ClosedTrade]) -> BacktestSummary:
    winning = sum(1 for trade in closed if trade.pnl > 0)
    losing = sum(1 for trade in closed if trade.pnl < 0)
    total_pnl = sum(trade.pnl for trade in closed)
    return BacktestSummary(
        trade_count=len(closed),
        winning_trades=winning,
        losing_trades=losing,
        total_pnl=total_pnl,
    )
