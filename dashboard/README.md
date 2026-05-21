# Dashboard

Localhost web app for tracking and setting up tradebot. Single-user, no auth,
binds to `127.0.0.1:8765` by default.

## Run

```bash
python -m orchestrator.main dashboard
# or
python -m orchestrator.main dashboard --port 8765
```

Open <http://127.0.0.1:8765/>.

The server refuses non-localhost binds unless `--i-know-what-im-doing` is
passed; even then, there is **no authentication** — anyone reaching the port
can flip the kill switch, flatten positions, edit config, or write the Kite
access token to `.env`.

## Stack

- FastAPI + uvicorn
- Jinja2 templates, server-rendered
- HTMX (CDN) for polling partials and form posts
- Tailwind CSS (CDN play script) for styling
- Chart.js (CDN) for the two equity-curve charts

No build step, no JS bundler. The whole frontend ships as `.html` files in
`dashboard/templates/` plus a 12-line `static/app.css`.

## Pages

- `/` **Overview** — Big kill-switch panel with toggle button. Today's
  realized PnL, open positions, signal/verdict/order counts. Last 5 journal
  events. Quick actions: Flatten All (with confirmation modal), Kite Login,
  Run Backtest. Polls `/_partials/overview/state` every 5s.
- `/live` **Live** — Open positions table (from `OrderStateStore`, ENTERED
  rows), recent 20 orders, daily PnL line chart (cumulative realized PnL
  from today's `order` journal events). Flatten All button with `FLATTEN`
  confirmation. Polls `/_partials/live/state` every 5s.
- `/orders` **Orders** — Filterable + paginated order table (state,
  symbol, per-page). Click an id to expand a JSON drawer. Admin buttons:
  `FAIL` / `CXL` mark an OPEN order as terminal in the store
  (`/api/orders/{id}/mark`); they do **not** cancel anything broker-side.
- `/backtests` **Backtests** — List of past runs (table from
  `backtest_runs` DuckDB). "Run new" form: strategy dropdown, symbol,
  bars count, qty, seed, EMA-crossover params. Submits to
  `POST /api/backtest/run` which runs the engine synchronously and
  redirects to the detail page.
- `/backtests/{run_id}` **Backtest detail** — Metrics table, Chart.js
  equity curve from the persisted JSON, closed-trades table.
- `/config` **Config** — YAML editor pre-filled with `config/config.yaml`
  (falls back to `config/config.example.yaml`). Validate button shows
  pydantic errors. Save button validates first, then writes the new file
  atomically and rotates the previous to `.bak`. Notice block warns that
  `.env` is off-limits.
- `/kite` **Kite Login** — Step 1: Kite login URL (built via
  `KiteConnect(api_key).login_url()`). Step 2: paste the request_token
  from the redirect. Step 3: `POST /api/kite/exchange` exchanges the
  token, writes `KITE_ACCESS_TOKEN=...` to `.env` (in-place line
  replacement; other lines preserved), and reloads settings. Banner shows
  current token presence + `.env` mtime + day-stale warning.
- `/journal` **Journal** — Live tail of the JSONL trading journal.
  Filters: event type (signal/verdict/risk_decision/order), symbol, limit.
  Pause/Resume toggle stops the 3s HTMX poller (uses the
  `[document.getElementById('pause-toggle').checked]` HTMX trigger
  expression).
- `/strategies` **Strategies** — Lists registered strategies, their
  timeframe, required indicators, doc line, and an enable/disable toggle
  persisted to `dashboard_strategy_settings`. Notice flags that
  orchestrator does not yet consume the enabled flag.

## API

All endpoints under `/api/*` return JSON.

| Method | Path                              | Body                                |
| ------ | --------------------------------- | ----------------------------------- |
| GET    | `/api/overview/state`             | —                                   |
| GET    | `/api/positions`                  | —                                   |
| GET    | `/api/orders?state=&symbol=&page=`| —                                   |
| GET    | `/api/journal/tail?since_ts=&...` | —                                   |
| POST   | `/api/kill/toggle`                | `{"enabled": bool}`                 |
| POST   | `/api/flatten`                    | `{"confirm": "FLATTEN"}`            |
| POST   | `/api/backtest/run`               | `{"strategy","symbol","params",...}`|
| POST   | `/api/config/validate`            | `{"yaml": "..."}`                   |
| POST   | `/api/config/save`                | `{"yaml": "..."}`                   |
| POST   | `/api/kite/exchange`              | `{"request_token": "..."}`          |
| POST   | `/api/orders/{id}/mark`           | `{"state": "FAILED"\|"CANCELLED"}`  |
| POST   | `/api/strategies/{id}/toggle`     | —                                   |

## Layout

```
dashboard/
├── README.md
├── __init__.py
├── server.py                 — create_app(), middleware, error handler
├── state.py                  — AppState singleton + DuckDB schemas
├── routes/
│   ├── __init__.py
│   ├── _common.py            — base context, service factories
│   ├── api.py                — JSON write/read endpoints
│   ├── overview.py
│   ├── live.py
│   ├── orders.py
│   ├── journal.py
│   ├── backtests.py
│   ├── config_ui.py
│   ├── kite_auth.py
│   └── strategies.py
├── services/
│   ├── __init__.py
│   ├── kill_switch.py        — touch/remove KILL file
│   ├── config_io.py          — validate / save YAML with .bak rotation
│   ├── journal_reader.py     — JSONL tail + filter + today PnL
│   ├── backtest_runner.py    — sync engine run + persist to DuckDB
│   ├── strategy_state.py     — enable/disable persistence
│   ├── kite_auth.py          — login URL + request_token exchange
│   └── orders.py             — filter, paginate, admin-mark
├── templates/
│   ├── base.html             — Tailwind + HTMX + Chart.js + kill banner
│   ├── error.html
│   ├── overview.html
│   ├── live.html
│   ├── orders.html
│   ├── journal.html
│   ├── backtests.html
│   ├── backtest_detail.html
│   ├── config.html
│   ├── kite.html
│   ├── strategies.html
│   └── partials/
│       ├── overview_state.html
│       ├── live_state.html
│       └── journal_rows.html
└── static/
    └── app.css
```

## Persistence

A separate DuckDB file at `<state_db dir>/dashboard.duckdb` holds two tables:

- `backtest_runs` — id, strategy, symbol, params (JSON), bars_count, run_at,
  total_pnl, sharpe, win_rate, mdd_pct, total_trades, result_json (full
  equity curve + closed trades + metrics)
- `strategy_settings` — strategy_id PRIMARY KEY, enabled BOOLEAN, updated_at

The `OrderStateStore` continues to own `order_state.duckdb`.

## TRADEOFFs

- **No auth**, single user, localhost bind by default. Loud banner on any
  non-localhost host. CLI refuses non-localhost unless
  `--i-know-what-im-doing` is set.
- **Backtest runner is synchronous** — `POST /api/backtest/run` blocks
  until the engine finishes. We wrap in `asyncio.to_thread` so the event
  loop stays responsive, but the user sees a brief UI freeze for big
  bar counts. Acceptable for personal use; future: a job queue.
- **JSON config editor saves raw text**, not a round-tripped YAML, so
  comments and key order are preserved. The cost is that we can't
  programmatically normalise the file.
- **Strategies enabled flag is informational in v1.** The orchestrator
  loop hard-codes its strategy. The DuckDB table is the future hook.
- **OrderStore-only positions view.** The Live page reads positions from
  `OrderStateStore` (ENTERED rows), not from a live Kite `positions()`
  call. A future iteration can split read paths between Paper / Kite
  brokers; for now this works for both because the order store is the
  canonical source.
- **Polling, not WebSockets.** The dashboard uses HTMX timed polls
  (5s overview/live, 3s journal). Simpler than WS, fine for personal
  volumes.
- **Admin order marks don't cancel broker-side.** Marking an order
  CANCELLED / FAILED only updates the local store. The dashboard surfaces
  this in tooltips; use `flatten` for real broker-side cancellation.
- **No CSRF.** All POSTs are unauthenticated; on localhost without a
  hostile process this is acceptable.

## Quality gates

```bash
ruff check .
mypy --strict .
pytest
```

All green. 259 tests; 51 added in this iteration (208 baseline preserved).
