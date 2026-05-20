from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.walk_forward import run_walk_forward, split_walk_forward_windows
from strategies.examples.ema_crossover import EmaCrossover

IST = ZoneInfo("Asia/Kolkata")


def test_split_walk_forward_windows() -> None:
    start = datetime(2024, 1, 1, tzinfo=IST)
    end = datetime(2024, 3, 1, tzinfo=IST)
    windows = split_walk_forward_windows(start, end, train_days=20, test_days=5)
    assert len(windows) >= 1
    assert windows[0].train_start == start


def test_run_walk_forward() -> None:
    timestamps = pd.date_range(
        "2024-01-01 09:15:00",
        periods=90,
        freq="1D",
        tz="Asia/Kolkata",
    )
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
            "symbol": "SYNTH",
        }
    )
    strategy = EmaCrossover(symbol="SYNTH")
    summary = run_walk_forward(
        strategy,
        bars,
        BacktestEngine(qty=1),
        train_days=30,
        test_days=7,
        step_days=7,
    )
    assert summary.windows
    assert isinstance(summary.total_pnl, float)
