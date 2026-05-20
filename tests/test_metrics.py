from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from backtest.engine import BacktestResult, BacktestSummary, ClosedTrade, EquityPoint
from backtest.metrics import (
    average_r,
    compute_performance_metrics,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate_pct,
)

IST = ZoneInfo("Asia/Kolkata")


def _ts(minute: int) -> datetime:
    return datetime(2024, 1, 1, 9, 15 + minute, tzinfo=IST)


def _trade(pnl: float, *, side: str = "LONG") -> ClosedTrade:
    entry = 100.0
    exit_price = entry + pnl if side == "LONG" else entry - pnl
    return ClosedTrade(
        symbol="X",
        side=side,  # type: ignore[arg-type]
        entry_price=entry,
        exit_price=exit_price,
        qty=1,
        entry_bar=0,
        exit_bar=1,
        pnl=pnl,
        exit_reason="target" if pnl > 0 else "stop_loss",
    )


def _equity_curve(values: list[float]) -> list[EquityPoint]:
    return [
        EquityPoint(bar_index=i, timestamp=_ts(i), equity=v)
        for i, v in enumerate(values)
    ]


def test_win_rate_and_profit_factor() -> None:
    trades = [_trade(10.0), _trade(-5.0), _trade(20.0), _trade(-5.0)]
    assert win_rate_pct(trades) == pytest.approx(50.0)
    assert profit_factor(trades) == pytest.approx(30.0 / 10.0)
    assert expectancy(trades) == pytest.approx(5.0)


def test_average_r_with_explicit_risks() -> None:
    trades = [_trade(20.0), _trade(-10.0)]
    risks = [10.0, 10.0]
    assert average_r(trades, trade_risks=risks) == pytest.approx(0.5)


def test_max_drawdown() -> None:
    curve = _equity_curve([0.0, 100.0, 50.0, 80.0])
    mdd, mdd_pct = max_drawdown(curve, initial_capital=1000.0)
    assert mdd == pytest.approx(50.0)
    assert mdd_pct == pytest.approx(50.0 / 1100.0 * 100.0)


def test_sharpe_known_returns() -> None:
    # Steady growth -> positive Sharpe
    equities = [0.0, 10.0, 20.0, 30.0]
    curve = _equity_curve(equities)
    sharpe = sharpe_ratio(curve, initial_capital=1000.0, periods_per_year=252)
    assert sharpe > 0


def test_sortino_no_downside() -> None:
    curve = _equity_curve([0.0, 1.0, 2.0, 3.0])
    assert sortino_ratio(curve, initial_capital=100.0, periods_per_year=252) == pytest.approx(
        float("inf")
    )


def test_compute_performance_metrics_synthetic() -> None:
    trades = [_trade(100.0), _trade(-50.0), _trade(100.0)]
    curve = _equity_curve([0.0, 100.0, 50.0, 150.0])
    result = BacktestResult(
        closed_trades=trades,
        equity_curve=curve,
        summary=BacktestSummary(
            trade_count=3,
            winning_trades=2,
            losing_trades=1,
            total_pnl=150.0,
        ),
    )
    metrics = compute_performance_metrics(
        result,
        initial_capital=10_000.0,
        trade_risks=[50.0, 50.0, 50.0],
        periods_per_year=4,
    )
    assert metrics.total_trades == 3
    assert metrics.total_pnl == 150.0
    assert metrics.win_rate_pct == pytest.approx(66.666, rel=0.01)
    assert metrics.average_r == pytest.approx((2.0 + -1.0 + 2.0) / 3)
    assert metrics.total_return_pct == pytest.approx(1.5, rel=0.01)
