# Backtest

Run historical simulations with `BacktestEngine().run(strategy, bars)` where `bars` is a DataFrame
or CSV path. Fills occur at the next bar open; stops/targets are checked on each bar's high/low.
Returns a typed `BacktestResult` with `closed_trades`, `equity_curve`, and `summary` stats.

## Performance metrics (Week 4)

```python
from backtest.engine import BacktestEngine
from backtest.metrics import compute_performance_metrics
from backtest.report import write_markdown_report, write_html_report

result = BacktestEngine(qty=1).run(strategy, bars)
metrics = compute_performance_metrics(result, initial_capital=100_000.0)
print(metrics.sharpe_ratio, metrics.max_drawdown_pct, metrics.profit_factor)

write_markdown_report("report.md", result, metrics=metrics)
write_html_report("report.html", result, metrics=metrics)
```

`compute_performance_metrics` exposes Sharpe, Sortino, max drawdown, win rate, average R,
expectancy, profit factor, and more. Pass `trade_risks` for accurate average R when stops are
not stored on closed trades.

## CLI

```bash
python -m orchestrator.main backtest --bars-count 1000 --report-md out.md --report-html out.html
```

See `engine.py` docstrings for documented TRADEOFFs (fill price vs signal entry, same-bar exits).
