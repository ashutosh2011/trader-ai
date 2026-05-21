# Tradebot

Production-grade algorithmic trading system for NSE: indicators, strategies, backtest, paper/live execution, risk, analyst co-decide, and journaling.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended)

## Install

```bash
cd tradebot
uv sync --extra dev
# or: python3 -m pip install -e ".[dev]"
```

## Config

```bash
cp config/config.example.yaml config/config.yaml
cp config/secrets.example.env .env
# Edit .env: KITE_API_KEY, KITE_ACCESS_TOKEN (for live), LLM keys (optional)
```

Kill switch: create `./KILL` or set `KILL_SWITCH=1`.

### Kite Connect (live trading)

1. Copy `config/secrets.example.env` to `.env` and set `KITE_API_KEY` and `KITE_API_SECRET`.
2. Register your Kite app redirect URL (e.g. `http://127.0.0.1` or `http://localhost`) in the [Kite developer console](https://developers.kite.trade/).
3. Generate a daily access token:

```bash
uv run python -m orchestrator.main kite-login
```

4. Open the printed login URL, sign in to Zerodha, and copy `request_token` from the redirect URL (or pass `--request-token`).
5. The CLI exchanges it for `KITE_ACCESS_TOKEN` and writes it to `.env`.

**Daily refresh:** Kite access tokens expire at the end of each trading day. Re-run `kite-login` every morning before `live --no-dry-run`.

Non-interactive (after you have the redirect token):

```bash
uv run python -m orchestrator.main kite-login --request-token YOUR_REQUEST_TOKEN
```

## Run tests

```bash
uv run pytest
uv run ruff check .
uv run mypy --strict .
```

With coverage:

```bash
uv run pytest --cov=core --cov=indicators --cov=strategies --cov=risk --cov=backtest \
  --cov=execution --cov=analyst --cov=orchestrator --cov=data --cov=journal --cov-report=term-missing
```

## CLI

```bash
uv run python -m orchestrator.main backtest --bars-count 1000
uv run python -m orchestrator.main paper --bars-count 500 --journal logs/paper.jsonl
uv run python -m orchestrator.main ab-test --bars-count 500
uv run python -m orchestrator.main live --dry-run   # paper on replay without Kite
uv run python -m orchestrator.main kite-login       # daily Kite token (see above)
```

**Live warnings:** `live --no-dry-run` with valid Kite credentials places real bracket orders. `--qty N` is a **HARD CEILING** applied after risk sizing (and re-floored to lot size); `--qty 0` aborts.

Safety rails:

- `live --no-dry-run` against synthetic bars is rejected outright. Use `--live-feed kite` (real ticks) or `--dry-run`.
- `live --no-dry-run --bars file.csv` is rejected unless `--allow-replay-live` is also passed. The CLI prints a loud `WARNING:` when allowed.
- `live --live-feed kite` requires `KITE_API_KEY` and `KITE_ACCESS_TOKEN`; missing credentials is a hard error.

`--live-feed kite` is currently the only real-tick source. Until it is wired into the production loop, `--no-dry-run` is effectively non-functional without `--allow-replay-live`; this is intentional.

## Custom indicators

Copy `indicators/custom/example_momentum.py` and register with `@register_indicator`. Import `indicators.custom` (or your module) so the registry loads it. See `indicators/custom/README.md`.

## Layout

```
tradebot/
├── core/           # Bar, Signal, Instrument (NSE EQ/F&O)
├── indicators/     # Builtin + custom registry
├── strategies/     # Strategy base + examples
├── backtest/       # Engine, metrics, walk-forward
├── data/           # Replay/live feeds, DuckDB store, Kite client
├── execution/      # Paper + Kite broker, reconciler
├── risk/           # Pre/post checks, kill switch
├── analyst/        # LLM co-decide layer
├── orchestrator/   # Loop, scheduler, CLI
├── journal/        # JSONL log + daily notebook
└── config/         # YAML + env settings
```

## EMA crossover defaults

| Parameter   | Default |
|------------|---------|
| Fast EMA   | 12      |
| Slow EMA   | 26      |
| ATR period | 14      |

Long on fast-above-slow cross; short on the opposite. Stop = entry ± 1×ATR, target = entry ± 2×ATR.
