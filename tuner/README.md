# Strategy tuner

Post-trade LLM review layer. Mirrors `screener/` and `analyst/`:

1. **Collect** closed trades from `backtest_runs.result_json` (lookback window).
2. **Prompt** the LLM with per-symbol stats (win rate, PnL, recent trades).
3. **Parse** a `TuningPlan` JSON (no code generation).
4. **Persist** pending recommendations; human **Apply** / **Reject** on the dashboard.

## Actions

| action | meaning |
|--------|---------|
| `keep` | No change |
| `modify_params` | Same strategy, new constructor params |
| `switch_strategy` | Replace with another registered strategy id |
| `disable` | Mark symbol disabled in `strategy_symbol_config` |

## CLI

```bash
python -m orchestrator.main tuner --provider stub
```

## Fallback

On timeout/parse errors → `DEFAULT_TUNING_PLAN` (empty recommendations, no changes).
