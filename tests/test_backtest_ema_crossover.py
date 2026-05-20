import pandas as pd

from backtest.engine import BacktestEngine
from strategies.examples.ema_crossover import EmaCrossover


def test_backtest_ema_crossover_runs_with_trades(synthetic_bars_1000: pd.DataFrame) -> None:
    strategy = EmaCrossover(symbol="SYNTH")
    engine = BacktestEngine(qty=1)
    result = engine.run(strategy, synthetic_bars_1000)
    assert result.trade_count >= 5
