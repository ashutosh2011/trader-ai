# Screener

The screener is a *read-only* watchlist generator. It asks an LLM for a
structured **filter formula**, evaluates that formula deterministically
against a configured universe of symbols, and persists the resulting
picks to DuckDB for review in the dashboard or via the CLI.

The screener does **not** wire into the orchestrator, risk manager, or
broker. Picks are advisory — the operator decides what to do with them.

## Quick start

```bash
# Offline smoke test (no API key, no network):
python -m orchestrator.main screener --provider stub

# View the result in the dashboard:
python -m orchestrator.main dashboard
# then open http://127.0.0.1:8765/screener
```

## Universe configuration

The universe lives at `config/universe.yaml` (user-local, gitignored).
If that file is missing, the loader falls back to
`config/universe.example.yaml` which ships with the repo.

```yaml
symbols:
  - {symbol: "RELIANCE", instrument_token: 738561, exchange: "NSE"}
  - {symbol: "INFY",     instrument_token: 408065, exchange: "NSE"}
```

* `symbol` — required, the trading symbol used both in the formula
  evaluation and the persisted pick row.
* `instrument_token` — optional. Required only for on-demand Kite
  candle fetch (`--fetch-missing`). Without it, the screener evaluates
  only symbols that already have bars in the local DuckDB candle store
  (run a backtest with the symbol first, or fill in the token from a
  Kite instruments dump).
* `exchange` — defaults to `"NSE"`. Informational; carried through to
  persistence and templates.

## Schema reference

The LLM emits a single JSON object matching `ScreenerFormula`
(`screener/schema.py`):

```jsonc
{
  "name": "Oversold reclaim",
  "timeframe": "day" | "5m",
  "side_bias": "long" | "short" | "both",
  "rationale": "human-readable reasoning",
  "filters": [Filter, ...]   // 1+ entries; all AND-combined
}
```

### Filter types

All filters evaluate against the **last closed bar** of each symbol.

* **`indicator`** — compares an indicator's latest value to either a
  constant or another indicator (one level of nesting).

  ```jsonc
  {"type": "indicator", "indicator": "rsi", "params": {"period": 14},
   "op": "<", "value": 35.0}

  {"type": "indicator", "indicator": "close", "op": ">",
   "compare_to": {"indicator": "sma", "params": {"period": 50}}}
  ```

  Exactly one of `value` or `compare_to` must be set.

* **`volume`** — tests last-bar volume against either an absolute
  threshold or a multiple of the rolling mean over `avg_window` bars.

  ```jsonc
  {"type": "volume", "op": ">", "value_x_avg": 1.5, "avg_window": 20}
  {"type": "volume", "op": ">", "value": 1000000}
  ```

  Exactly one of `value` or `value_x_avg` must be set.

* **`price_change`** — percent change over the last `window` closed
  bars.

  ```jsonc
  {"type": "price_change", "window": 5, "op": ">", "value_pct": 3.0}
  ```

Allowed comparison ops: `>`, `<`, `>=`, `<=`, `==`.

## Indicator whitelist

The screener only honours these LLM-facing indicator names (see
`screener/registry.py`):

| Name              | Description                                | Common params           |
| ----------------- | ------------------------------------------ | ----------------------- |
| `rsi`             | Wilder RSI                                 | `period` (int, def 14)  |
| `sma`             | Simple moving average of close             | `period` (int)          |
| `ema`             | Exponential moving average of close        | `span` (int)            |
| `atr`             | Wilder ATR                                 | `period` (int, def 14)  |
| `close` / `open` / `high` / `low` | Latest OHLC value          | —                       |
| `macd_line` / `macd_signal` / `macd_hist` | MACD components    | `fast`, `slow`, `signal` |
| `bb_upper` / `bb_middle` / `bb_lower` | Bollinger Bands        | `period`, `mult`         |
| `supertrend_dir`  | Supertrend direction (+1 up, -1 down)      | `period`, `multiplier`   |

Any other name → parse error → fallback to `DEFAULT_FORMULA`.

## Example formulas

### Long mean-revert (daily)

```json
{
  "name": "Oversold reclaim",
  "timeframe": "day",
  "side_bias": "long",
  "rationale": "Range-bound regime; buy oversold dips above the 50-day SMA.",
  "filters": [
    {"type": "indicator", "indicator": "rsi", "params": {"period": 14}, "op": "<", "value": 35.0},
    {"type": "indicator", "indicator": "close", "op": ">",
     "compare_to": {"indicator": "sma", "params": {"period": 50}}},
    {"type": "volume", "op": ">", "value_x_avg": 1.2, "avg_window": 20}
  ]
}
```

### Short breakdown (5m)

```json
{
  "name": "5m breakdown short",
  "timeframe": "5m",
  "side_bias": "short",
  "rationale": "Downtrend regime; short stocks losing the 20-period EMA on volume.",
  "filters": [
    {"type": "indicator", "indicator": "close", "op": "<",
     "compare_to": {"indicator": "ema", "params": {"span": 20}}},
    {"type": "price_change", "window": 5, "op": "<", "value_pct": -1.5},
    {"type": "volume", "op": ">", "value_x_avg": 1.5, "avg_window": 20}
  ]
}
```

## Fallback behaviour

When the LLM call fails the runner returns `DEFAULT_FORMULA`
(`screener/llm_screener.py`) and persists a run with one of:

| Status                    | Meaning                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| `ok`                      | Provider returned a valid formula.                                 |
| `fallback_transport`      | Timeout / `httpx.HTTPError`. Default formula used.                 |
| `fallback_parse_error`    | JSON / schema validation failed. Default formula used.             |
| `fallback_unexpected`     | Unhandled exception. Default formula used.                         |

The runner **never crashes** — even with no LLM keys, you get a record
with a status badge explaining what went wrong.

## CLI flags

```text
--universe PATH         Universe YAML (default: config/universe.yaml).
--provider stub|...     stub uses DEFAULT_FORMULA (offline). Others need keys.
--fetch-missing         Fetch missing bars via Kite (requires credentials).
--bars-back N           Recent bars to load per symbol (default 200).
--output table|json     Output format (default: table).
--notes TEXT            Free-form market-context notes.
--index-summary TEXT    Short regime hint passed to the LLM.
--dashboard-db PATH     Override the DuckDB file (default: dashboard.duckdb).
```
