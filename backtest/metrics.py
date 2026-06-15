"""Performance metrics computed from backtest trades and equity curve."""

from collections.abc import Sequence
from math import sqrt

import numpy as np
from pydantic import BaseModel, ConfigDict

from backtest.engine import BacktestResult, ClosedTrade, EquityPoint

DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_PERIODS_PER_YEAR = 252 * 375  # TRADEOFF: 1m bars, ~375 min/session on NSE
TRADING_DAYS_PER_YEAR = 252
NSE_MINUTES_PER_SESSION = 375  # 09:15–15:30 IST

# Annualization factors keyed by Kite interval. Using the wrong factor (the
# old code always assumed 1-minute bars) over- or under-states Sharpe/Sortino
# by orders of magnitude on coarser intervals, so the runner now passes the
# real timeframe through.
_BARS_PER_SESSION: dict[str, int] = {
    "minute": NSE_MINUTES_PER_SESSION,
    "3minute": 125,
    "5minute": 75,
    "10minute": 38,
    "15minute": 25,
    "30minute": 13,
    "60minute": 6,
    "day": 1,
}


def periods_per_year_for_timeframe(timeframe: str | None) -> int:
    """Return the annualization factor for a Kite ``timeframe``.

    Falls back to the 1-minute default when the interval is unknown or
    ``None`` so legacy callers keep their previous behaviour.
    """
    if timeframe is None:
        return DEFAULT_PERIODS_PER_YEAR
    bars = _BARS_PER_SESSION.get(timeframe.strip().lower())
    if bars is None:
        return DEFAULT_PERIODS_PER_YEAR
    return TRADING_DAYS_PER_YEAR * bars


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
    # Extended analytics (default to neutral values so any future caller that
    # constructs the model without them stays valid).
    cagr_pct: float = 0.0
    calmar_ratio: float = 0.0
    exposure_pct: float = 0.0
    avg_trade_duration_bars: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    longest_drawdown_bars: int = 0
    total_fees: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0


def compute_performance_metrics(
    result: BacktestResult,
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
    timeframe: str | None = None,
    trade_risks: Sequence[float] | None = None,
    benchmark_prices: Sequence[float] | None = None,
    total_bars: int | None = None,
) -> PerformanceMetrics:
    """Compute performance metrics from a :class:`BacktestResult`.

    Args:
        result: Backtest output with trades and equity curve.
        initial_capital: Starting equity for return and drawdown math.
        risk_free_rate: Annualized risk-free rate for Sharpe/Sortino excess returns.
        periods_per_year: Explicit annualization factor. When ``None`` it is
            derived from ``timeframe`` (defaulting to 1-minute bars).
        timeframe: Kite interval (e.g. ``"5minute"``) used to pick the
            annualization factor when ``periods_per_year`` is not given.
        trade_risks: Optional per-trade initial risk (|entry - stop| * qty). Required
            for meaningful average R when stops are not stored on closed trades.
        benchmark_prices: Optional close-price series for a buy-and-hold
            benchmark; enables ``benchmark_return_pct`` and ``alpha_pct``.
        total_bars: Bar count used as the exposure denominator; defaults to the
            length of the equity curve.

    Returns:
        Frozen :class:`PerformanceMetrics` snapshot.
    """
    trades = result.closed_trades
    if periods_per_year is None:
        periods_per_year = periods_per_year_for_timeframe(timeframe)
    bar_total = total_bars if total_bars is not None else len(result.equity_curve)
    mdd, mdd_pct = max_drawdown(result.equity_curve, initial_capital=initial_capital)
    cagr = cagr_pct(result.equity_curve, initial_capital=initial_capital)
    benchmark = benchmark_return_pct(benchmark_prices)
    total_return = total_return_pct(result.equity_curve, initial_capital=initial_capital)
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
        max_drawdown=mdd,
        max_drawdown_pct=mdd_pct,
        win_rate_pct=win_rate_pct(trades),
        average_r=average_r(trades, trade_risks=trade_risks),
        expectancy=expectancy(trades),
        profit_factor=profit_factor(trades),
        total_return_pct=total_return,
        total_trades=len(trades),
        avg_win=avg_win(trades),
        avg_loss=avg_loss(trades),
        total_pnl=result.total_pnl,
        winning_trades=sum(1 for t in trades if t.pnl > 0),
        losing_trades=sum(1 for t in trades if t.pnl < 0),
        cagr_pct=cagr,
        calmar_ratio=calmar_ratio(cagr, mdd_pct),
        exposure_pct=exposure_pct(trades, bar_total),
        avg_trade_duration_bars=avg_trade_duration_bars(trades),
        max_consecutive_wins=max_consecutive(trades, winning=True),
        max_consecutive_losses=max_consecutive(trades, winning=False),
        best_trade=max((t.pnl for t in trades), default=0.0),
        worst_trade=min((t.pnl for t in trades), default=0.0),
        gross_profit=gross_profit(trades),
        gross_loss=gross_loss(trades),
        longest_drawdown_bars=longest_drawdown_bars(
            result.equity_curve, initial_capital=initial_capital
        ),
        total_fees=float(sum(t.fees for t in trades)),
        benchmark_return_pct=benchmark,
        alpha_pct=total_return - benchmark,
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


def years_elapsed(equity_curve: Sequence[EquityPoint]) -> float:
    """Calendar years between the first and last equity timestamp."""
    if len(equity_curve) < 2:
        return 0.0
    span = equity_curve[-1].timestamp - equity_curve[0].timestamp
    return span.total_seconds() / (365.25 * 24 * 3600)


def cagr_pct(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> float:
    """Compound annual growth rate (%) of portfolio value.

    Falls back to the simple total return when the span is under a day or the
    final value is non-positive (CAGR is undefined for a wiped-out account).
    """
    values = portfolio_values(equity_curve, initial_capital=initial_capital)
    if len(values) < 2 or initial_capital <= 0:
        return 0.0
    final_value = float(values[-1])
    years = years_elapsed(equity_curve)
    if years < (1.0 / 365.25) or final_value <= 0:
        return total_return_pct(equity_curve, initial_capital=initial_capital)
    return float(((final_value / initial_capital) ** (1.0 / years) - 1.0) * 100.0)


def calmar_ratio(cagr: float, max_drawdown_percent: float) -> float:
    """CAGR divided by absolute max drawdown %. Zero when no drawdown."""
    if max_drawdown_percent == 0.0:
        return 0.0
    return cagr / abs(max_drawdown_percent)


def gross_profit(trades: Sequence[ClosedTrade]) -> float:
    """Sum of positive trade PnL."""
    return float(sum(t.pnl for t in trades if t.pnl > 0))


def gross_loss(trades: Sequence[ClosedTrade]) -> float:
    """Absolute sum of negative trade PnL (>= 0)."""
    return float(abs(sum(t.pnl for t in trades if t.pnl < 0)))


def exposure_pct(trades: Sequence[ClosedTrade], total_bars: int) -> float:
    """Fraction of bars (%) spent holding a position.

    Each trade occupies ``exit_bar - entry_bar + 1`` bars. Overlapping bars
    are not double-counted because the engine holds at most one position at a
    time. Capped at 100%.
    """
    if total_bars <= 0 or not trades:
        return 0.0
    held = sum(max(0, t.exit_bar - t.entry_bar) + 1 for t in trades)
    return float(min(100.0, held / total_bars * 100.0))


def avg_trade_duration_bars(trades: Sequence[ClosedTrade]) -> float:
    """Mean number of bars between entry and exit across trades."""
    if not trades:
        return 0.0
    durations = [max(0, t.exit_bar - t.entry_bar) for t in trades]
    return float(np.mean(durations))


def max_consecutive(trades: Sequence[ClosedTrade], *, winning: bool) -> int:
    """Longest run of consecutive winning (or losing) trades."""
    best = 0
    current = 0
    for trade in trades:
        is_win = trade.pnl > 0
        is_loss = trade.pnl < 0
        match = is_win if winning else is_loss
        if match:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def drawdown_series(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> list[float]:
    """Per-bar drawdown (%) from the running peak — the 'underwater' curve."""
    values = portfolio_values(equity_curve, initial_capital=initial_capital)
    out: list[float] = []
    peak = float(values[0]) if len(values) else initial_capital
    for value in values:
        v = float(value)
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100.0 if peak > 0 else 0.0
        out.append(-dd)
    return out


def longest_drawdown_bars(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> int:
    """Longest stretch of bars (count) spent below a prior equity peak."""
    values = portfolio_values(equity_curve, initial_capital=initial_capital)
    if len(values) == 0:
        return 0
    peak = float(values[0])
    longest = 0
    current = 0
    for value in values:
        v = float(value)
        if v >= peak:
            peak = v
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def benchmark_return_pct(prices: Sequence[float] | None) -> float:
    """Buy-and-hold return (%) of the first→last close of ``prices``."""
    if prices is None or len(prices) < 2:
        return 0.0
    first = float(prices[0])
    last = float(prices[-1])
    if first <= 0:
        return 0.0
    return float((last - first) / first * 100.0)


def monthly_returns(
    equity_curve: Sequence[EquityPoint],
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> list[dict[str, float | int | str]]:
    """Calendar-month returns (%) derived from month-end portfolio value.

    Returns a list of ``{"month": "YYYY-MM", "year": int, "month_num": int,
    "return_pct": float}`` ordered chronologically. Each month's return is
    measured against the previous month-end value (or initial capital for the
    first month), so the series compounds back to the total return.
    """
    if not equity_curve:
        return []
    month_end: dict[str, float] = {}
    month_meta: dict[str, tuple[int, int]] = {}
    for point in equity_curve:
        key = f"{point.timestamp.year:04d}-{point.timestamp.month:02d}"
        month_end[key] = initial_capital + point.equity
        month_meta[key] = (point.timestamp.year, point.timestamp.month)
    out: list[dict[str, float | int | str]] = []
    prev_value = initial_capital
    for key in sorted(month_end):
        end_value = month_end[key]
        ret = (end_value - prev_value) / prev_value * 100.0 if prev_value != 0 else 0.0
        year, month_num = month_meta[key]
        out.append(
            {
                "month": key,
                "year": year,
                "month_num": month_num,
                "return_pct": float(ret),
            }
        )
        prev_value = end_value
    return out
