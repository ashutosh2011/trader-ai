"""Walk-forward backtest windowing and aggregation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from backtest.engine import BacktestEngine, BacktestResult
    from strategies.base import Strategy

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)


class WalkForwardWindow(BaseModel):
    """Single train/test date window."""

    model_config = ConfigDict(frozen=True)

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


class WalkForwardWindowResult(BaseModel):
    """Backtest metrics for one walk-forward test window."""

    model_config = ConfigDict(frozen=True)

    window: WalkForwardWindow
    trade_count: int
    total_pnl: float
    win_rate_pct: float


class WalkForwardSummary(BaseModel):
    """Aggregated walk-forward results."""

    model_config = ConfigDict(frozen=True)

    windows: list[WalkForwardWindowResult]
    total_pnl: float
    avg_pnl_per_window: float
    avg_win_rate_pct: float


def split_walk_forward_windows(
    start: datetime,
    end: datetime,
    *,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[WalkForwardWindow]:
    """Split a date range into rolling train/test windows."""
    if train_days < 1 or test_days < 1:
        msg = "train_days and test_days must be >= 1"
        raise ValueError(msg)
    step = step_days if step_days is not None else test_days
    start_ts = pd.Timestamp(start).tz_convert(IST)
    end_ts = pd.Timestamp(end).tz_convert(IST)
    windows: list[WalkForwardWindow] = []
    cursor = start_ts
    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > end_ts:
            break
        windows.append(
            WalkForwardWindow(
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                test_start=test_start.to_pydatetime(),
                test_end=test_end.to_pydatetime(),
            )
        )
        cursor = cursor + pd.Timedelta(days=step)
    return windows


def slice_bars_by_window(
    bars: pd.DataFrame,
    window: WalkForwardWindow,
    *,
    phase: str,
) -> pd.DataFrame:
    """Extract bars for train or test phase of a window."""
    if phase not in {"train", "test"}:
        msg = "phase must be 'train' or 'test'"
        raise ValueError(msg)
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(IST)
    if phase == "train":
        start, end = window.train_start, window.train_end
    else:
        start, end = window.test_start, window.test_end
    mask = (frame["timestamp"] >= pd.Timestamp(start)) & (frame["timestamp"] < pd.Timestamp(end))
    return frame.loc[mask].reset_index(drop=True)


def run_walk_forward(
    strategy: Strategy,
    bars: pd.DataFrame,
    engine: BacktestEngine,
    *,
    train_days: int = 30,
    test_days: int = 7,
    step_days: int | None = None,
) -> WalkForwardSummary:
    """Run backtests on each test window and aggregate results."""
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(IST)
    start = frame["timestamp"].min().to_pydatetime()
    end = frame["timestamp"].max().to_pydatetime()
    windows = split_walk_forward_windows(
        start, end, train_days=train_days, test_days=test_days, step_days=step_days
    )
    results: list[WalkForwardWindowResult] = []
    for window in windows:
        test_bars = slice_bars_by_window(frame, window, phase="test")
        if test_bars.empty:
            continue
        outcome: BacktestResult = engine.run(strategy, test_bars)
        wins = sum(1 for t in outcome.closed_trades if t.pnl > 0)
        win_rate = (wins / outcome.trade_count * 100.0) if outcome.trade_count else 0.0
        results.append(
            WalkForwardWindowResult(
                window=window,
                trade_count=outcome.trade_count,
                total_pnl=outcome.total_pnl,
                win_rate_pct=win_rate,
            )
        )
    total_pnl = sum(r.total_pnl for r in results)
    count = len(results)
    avg_pnl = total_pnl / count if count else 0.0
    avg_win = sum(r.win_rate_pct for r in results) / count if count else 0.0
    logger.info("walk_forward_complete", windows=count, total_pnl=total_pnl)
    return WalkForwardSummary(
        windows=results,
        total_pnl=total_pnl,
        avg_pnl_per_window=avg_pnl,
        avg_win_rate_pct=avg_win,
    )
