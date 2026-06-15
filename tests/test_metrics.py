from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from backtest.engine import BacktestResult, BacktestSummary, ClosedTrade, EquityPoint
from backtest.metrics import (
    DEFAULT_PERIODS_PER_YEAR,
    average_r,
    benchmark_return_pct,
    cagr_pct,
    calmar_ratio,
    compute_performance_metrics,
    drawdown_series,
    expectancy,
    exposure_pct,
    longest_drawdown_bars,
    max_consecutive,
    max_drawdown,
    monthly_returns,
    periods_per_year_for_timeframe,
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


def test_periods_per_year_for_timeframe() -> None:
    assert periods_per_year_for_timeframe("day") == 252
    assert periods_per_year_for_timeframe("5minute") == 252 * 75
    assert periods_per_year_for_timeframe("minute") == DEFAULT_PERIODS_PER_YEAR
    # Unknown / None falls back to the 1-minute default.
    assert periods_per_year_for_timeframe(None) == DEFAULT_PERIODS_PER_YEAR
    assert periods_per_year_for_timeframe("bogus") == DEFAULT_PERIODS_PER_YEAR


def test_cagr_and_calmar() -> None:
    # One year of daily-ish points, doubling capital → ~100% CAGR.
    curve = [
        EquityPoint(bar_index=0, timestamp=datetime(2023, 1, 1, tzinfo=IST), equity=0.0),
        EquityPoint(bar_index=1, timestamp=datetime(2024, 1, 1, tzinfo=IST), equity=100.0),
    ]
    cagr = cagr_pct(curve, initial_capital=100.0)
    assert cagr == pytest.approx(100.0, rel=0.02)
    assert calmar_ratio(cagr, 50.0) == pytest.approx(cagr / 50.0)
    assert calmar_ratio(cagr, 0.0) == 0.0


def test_exposure_and_streaks() -> None:
    trades = [_trade(10.0), _trade(-5.0), _trade(-5.0), _trade(20.0)]
    # Each _trade spans bars 0->1 == 2 bars held; 4 trades -> 8 of 20 bars.
    assert exposure_pct(trades, 20) == pytest.approx(40.0)
    assert exposure_pct(trades, 0) == 0.0
    assert max_consecutive(trades, winning=False) == 2
    assert max_consecutive(trades, winning=True) == 1


def test_benchmark_return_pct() -> None:
    assert benchmark_return_pct([100.0, 110.0]) == pytest.approx(10.0)
    assert benchmark_return_pct([100.0]) == 0.0
    assert benchmark_return_pct(None) == 0.0


def test_drawdown_series_and_longest() -> None:
    curve = _equity_curve([0.0, 100.0, 50.0, 50.0, 200.0])
    series = drawdown_series(curve, initial_capital=1000.0)
    assert len(series) == len(curve)
    # All points are <= 0 (underwater), peak point is exactly 0.
    assert max(series) == pytest.approx(0.0)
    assert min(series) < 0.0
    assert longest_drawdown_bars(curve, initial_capital=1000.0) == 2


def test_monthly_returns() -> None:
    curve = [
        EquityPoint(bar_index=0, timestamp=datetime(2024, 1, 31, tzinfo=IST), equity=100.0),
        EquityPoint(bar_index=1, timestamp=datetime(2024, 2, 29, tzinfo=IST), equity=300.0),
    ]
    rows = monthly_returns(curve, initial_capital=1000.0)
    assert [r["month"] for r in rows] == ["2024-01", "2024-02"]
    # Jan: 1100/1000 - 1 = 10%; Feb: 1300/1100 - 1 ≈ 18.18%.
    assert rows[0]["return_pct"] == pytest.approx(10.0)
    assert rows[1]["return_pct"] == pytest.approx(1300.0 / 1100.0 * 100.0 - 100.0)


def test_compute_metrics_includes_benchmark_alpha() -> None:
    trades = [_trade(100.0)]
    curve = _equity_curve([0.0, 100.0])
    result = BacktestResult(
        closed_trades=trades,
        equity_curve=curve,
        summary=BacktestSummary(
            trade_count=1, winning_trades=1, losing_trades=0, total_pnl=100.0
        ),
    )
    metrics = compute_performance_metrics(
        result,
        initial_capital=10_000.0,
        timeframe="day",
        benchmark_prices=[100.0, 105.0],
    )
    assert metrics.benchmark_return_pct == pytest.approx(5.0)
    assert metrics.alpha_pct == pytest.approx(metrics.total_return_pct - 5.0)
    assert metrics.best_trade == pytest.approx(100.0)
