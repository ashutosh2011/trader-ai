import pandas as pd

import strategies  # noqa: F401 — register examples
from backtest.engine import BacktestEngine
from strategies.examples.bbands_breakout import BBandsBreakout
from strategies.examples.macd_trend import MacdTrend
from strategies.examples.rsi_mean_revert import RsiMeanRevert
from strategies.examples.supertrend_follow import SupertrendFollow


def test_backtest_rsi_mean_revert_runs(synthetic_bars_1000: pd.DataFrame) -> None:
    strategy = RsiMeanRevert(symbol="SYNTH")
    engine = BacktestEngine(qty=1)
    result = engine.run(strategy, synthetic_bars_1000)
    assert len(result.closed_trades) >= 0


def test_backtest_bbands_breakout_runs(synthetic_bars_1000: pd.DataFrame) -> None:
    strategy = BBandsBreakout(symbol="SYNTH")
    engine = BacktestEngine(qty=1)
    result = engine.run(strategy, synthetic_bars_1000)
    assert len(result.closed_trades) >= 0


def test_backtest_macd_trend_runs(synthetic_bars_1000: pd.DataFrame) -> None:
    strategy = MacdTrend(symbol="SYNTH")
    engine = BacktestEngine(qty=1)
    result = engine.run(strategy, synthetic_bars_1000)
    assert len(result.closed_trades) >= 0


def test_backtest_supertrend_follow_runs_with_trades(
    synthetic_bars_1000: pd.DataFrame,
) -> None:
    # Supertrend on the trending synthetic bars reliably produces flips; we use
    # it as the smoke test that signals make it through the engine end-to-end.
    strategy = SupertrendFollow(symbol="SYNTH")
    engine = BacktestEngine(qty=1)
    result = engine.run(strategy, synthetic_bars_1000)
    assert result.trade_count > 0
