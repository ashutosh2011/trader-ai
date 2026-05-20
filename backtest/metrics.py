"""Performance metrics computed from backtest trades and equity curve."""

from collections.abc import Sequence
from math import sqrt

import numpy as np
from pydantic import BaseModel, ConfigDict

from backtest.engine import BacktestResult, ClosedTrade, EquityPoint

DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_PERIODS_PER_YEAR = 252 * 375  # TRADEOFF: 1m bars, ~375 min/session on NSE


class PerformanceMetrics(BaseModel):
    """Aggregated backtest performance statistics."""

    model_config = ConfigDict(frozen=True)

    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate_pct: float
    average_r: float
    expectancy: float
    profit_factor: float
    total_return_pct: float
    total_trades: int
    avg_win: float
    avg_loss: float
    total_pnl: float
    winning_trades: int
    losing_trades: int


def compute_performance_metrics(
    result: BacktestResult,
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    trade_risks: Sequence[float] | None = None,
) -> PerformanceMetrics:
    """Compute performance metrics from a :class:`BacktestResult`.

    Args:
        result: Backtest output with trades and equity curve.
        initial_capital: Starting equity for return and drawdown math.
        risk_free_rate: Annualized risk-free rate for Sharpe/Sortino excess returns.
        periods_per_year: Annualization factor for per-bar equity returns.
        trade_risks: Optional per-trade initial risk (|entry - stop| * qty). Required
            for meaningful average R when stops are not stored on closed trades.

    Returns:
        Frozen :class:`PerformanceMetrics` snapshot.
    """
    trades = result.closed_trades
    return PerformanceMetrics(
        sharpe_ratio=sharpe_ratio(
            result.equity_curve,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        sortino_ratio=sortino_ratio(
            result.equity_curve,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        max_drawdown=max_drawdown(result.equity_curve, initial_capital=initial_capital)[0],
        max_drawdown_pct=max_drawdown(result.equity_curve, initial_capital=initial_capital)[1],
        win_rate_pct=win_rate_pct(trades),
        average_r=average_r(trades, trade_risks=trade_risks),
        expectancy=expectancy(trades),
        profit_factor=profit_factor(trades),
        total_return_pct=total_return_pct(result.equity_curve, initial_capital=initial_capital),
        total_trades=len(trades),
        avg_win=avg_win(trades),
        avg_loss=avg_loss(trades),
        total_pnl=result.total_pnl,
        winning_trades=sum(1 for t in trades if t.pnl > 0),
        losing_trades=sum(1 for t in trades if t.pnl < 0),
    )


def portfolio_values(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float,
) -> np.ndarray:
    """Mark-to-market portfolio value series (initial capital + realized PnL)."""
    if not equity_curve:
        return np.array([initial_capital], dtype=float)
    return initial_capital + np.array([p.equity for p in equity_curve], dtype=float)


def per_period_returns(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float,
) -> np.ndarray:
    """Per-bar simple returns from the equity curve."""
    values = portfolio_values(equity_curve, initial_capital=initial_capital)
    if len(values) < 2:
        return np.array([], dtype=float)
    prev = values[:-1]
    prev = np.where(prev == 0, np.nan, prev)
    returns: np.ndarray = np.diff(values) / prev
    filtered: np.ndarray = returns[np.isfinite(returns)]
    return filtered


def sharpe_ratio(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio from equity-curve per-bar returns."""
    returns = per_period_returns(equity_curve, initial_capital=initial_capital)
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(excess) / std * sqrt(periods_per_year))


def sortino_ratio(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    returns = per_period_returns(equity_curve, initial_capital=initial_capital)
    if len(returns) < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if float(np.mean(excess)) > 0 else 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0.0:
        return 0.0
    return float(np.mean(excess) / downside_std * sqrt(periods_per_year))


def max_drawdown(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> tuple[float, float]:
    """Return (absolute MDD, MDD %) from the equity curve."""
    values = portfolio_values(equity_curve, initial_capital=initial_capital)
    if len(values) == 0:
        return 0.0, 0.0
    peak = values[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for value in values:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown / peak if peak > 0 else 0.0
    return float(max_dd), float(max_dd_pct * 100.0)


def win_rate_pct(trades: Sequence[ClosedTrade]) -> float:
    """Percentage of trades with positive PnL."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl > 0)
    return 100.0 * wins / len(trades)


def avg_win(trades: Sequence[ClosedTrade]) -> float:
    """Mean PnL of winning trades (0 if none)."""
    wins = [t.pnl for t in trades if t.pnl > 0]
    return float(np.mean(wins)) if wins else 0.0


def avg_loss(trades: Sequence[ClosedTrade]) -> float:
    """Mean PnL of losing trades (negative or 0 if none)."""
    losses = [t.pnl for t in trades if t.pnl < 0]
    return float(np.mean(losses)) if losses else 0.0


def profit_factor(trades: Sequence[ClosedTrade]) -> float:
    """Gross profit divided by absolute gross loss."""
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def expectancy(trades: Sequence[ClosedTrade]) -> float:
    """Per-trade expectancy: win_rate * avg_win + loss_rate * avg_loss."""
    if not trades:
        return 0.0
    n = len(trades)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    win_rate = len(wins) / n
    loss_rate = len(losses) / n
    mean_win = float(np.mean(wins)) if wins else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return win_rate * mean_win + loss_rate * mean_loss


def average_r(
    trades: Sequence[ClosedTrade],
    *,
    trade_risks: Sequence[float] | None = None,
) -> float:
    """Average R-multiple across trades.

    R = pnl / initial_risk per trade. When ``trade_risks`` is omitted, initial
    risk is approximated as ``abs(pnl)`` for losers and ``abs(pnl)/2`` for
    winners (TRADEOFF: coarse proxy without stop prices on closed trades).
    """
    if not trades:
        return 0.0
    r_values: list[float] = []
    for i, trade in enumerate(trades):
        if trade_risks is not None:
            risk = trade_risks[i]
        elif trade.pnl < 0:
            risk = abs(trade.pnl)
        elif trade.pnl > 0:
            risk = abs(trade.pnl) / 2.0
        else:
            risk = 1.0
        if risk <= 0:
            continue
        r_values.append(trade.pnl / risk)
    return float(np.mean(r_values)) if r_values else 0.0


def total_return_pct(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> float:
    """Total return as a percentage of initial capital."""
    values = portfolio_values(equity_curve, initial_capital=initial_capital)
    if len(values) == 0 or initial_capital == 0:
        return 0.0
    return float((values[-1] - initial_capital) / initial_capital * 100.0)
