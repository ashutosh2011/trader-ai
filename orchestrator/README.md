# Orchestrator

Co-decide loop (strategy → risk → analyst → broker), market scheduler, and CLI.

```bash
uv run python -m orchestrator.main backtest --bars-count 1000
uv run python -m orchestrator.main paper --journal logs/paper.jsonl
uv run python -m orchestrator.main live --dry-run
```

`MarketScheduler` gates bars to NSE hours (09:15–15:30 IST), holidays, and F&O expiry rules.
