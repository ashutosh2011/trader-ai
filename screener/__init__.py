"""LLM-driven screener layer.

The screener produces a *read-only* watchlist by asking an LLM for a
structured filter formula, evaluating that formula against a universe of
symbols using locally-stored OHLCV candles, and persisting the resulting
picks to DuckDB. It is intentionally decoupled from the orchestrator/auto-
trade pipeline; nothing in this package places real orders or modifies
risk/position state.

Key entry points:
    * :class:`screener.runner.ScreenerRunner` — orchestrates a full run.
    * :class:`screener.llm_screener.LLMScreener` — LLM-facing layer with
      the analyst-style fallback ladder (timeout → parse → unexpected).
    * :func:`screener.evaluator.evaluate` — pure evaluation against a dict
      of candle DataFrames.
"""
