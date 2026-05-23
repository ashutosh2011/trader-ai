"""Tests for the parameter-sweep flow."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest
from fastapi.testclient import TestClient

from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.services.backtest_runner import BacktestRunner
from dashboard.services.instruments import InstrumentsService
from dashboard.services.sweep_runner import (
    MAX_SWEEP_CELLS,
    BarsLoadResult,
    SweepCell,
    SweepConfig,
    SweepRunner,
    expand,
)
from dashboard.state import (
    BACKTEST_GROUPS_SCHEMA,
    BACKTEST_RUNS_SCHEMA,
    BACKTEST_SWEEPS_SCHEMA,
    INSTRUMENTS_META_SCHEMA,
    INSTRUMENTS_SCHEMA,
    STRATEGY_SETTINGS_SCHEMA,
    AppState,
    _add_optional_column,
    set_app_state,
)
from data.synthetic import make_synthetic_bars


def _instruments_fetcher(_settings: AppSettings, _exchange: str) -> list[dict[str, Any]]:
    return list(_INSTRUMENTS)


_INSTRUMENTS: list[dict[str, Any]] = [
    {
        "instrument_token": 738561,
        "exchange_token": 0,
        "tradingsymbol": "RELIANCE",
        "name": "Reliance Industries",
        "last_price": 2500.0,
        "expiry": "",
        "strike": 0,
        "tick_size": 0.05,
        "lot_size": 1,
        "instrument_type": "EQ",
        "segment": "NSE",
        "exchange": "NSE",
    },
    {
        "instrument_token": 408065,
        "exchange_token": 0,
        "tradingsymbol": "INFY",
        "name": "Infosys",
        "last_price": 1500.0,
        "expiry": "",
        "strike": 0,
        "tick_size": 0.05,
        "lot_size": 1,
        "instrument_type": "EQ",
        "segment": "NSE",
        "exchange": "NSE",
    },
]


def _from_to() -> tuple[datetime, datetime]:
    end = datetime(2024, 1, 5, 15, 30, tzinfo=UTC)
    start = end - timedelta(days=2)
    return start, end


def _build_dashboard_conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "dash.duckdb"))
    conn.execute(BACKTEST_RUNS_SCHEMA)
    conn.execute(BACKTEST_GROUPS_SCHEMA)
    conn.execute(BACKTEST_SWEEPS_SCHEMA)
    _add_optional_column(
        conn, "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS group_id VARCHAR"
    )
    _add_optional_column(
        conn, "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS sweep_id VARCHAR"
    )
    conn.execute(STRATEGY_SETTINGS_SCHEMA)
    conn.execute(INSTRUMENTS_SCHEMA)
    conn.execute(INSTRUMENTS_META_SCHEMA)
    return conn


def _populate_instruments(
    conn: duckdb.DuckDBPyConnection, settings: AppSettings
) -> InstrumentsService:
    service = InstrumentsService(conn, settings=settings, fetcher=_instruments_fetcher)
    service.ensure_schema()
    service.refresh()
    return service


# ---------------------------------------------------------------------------
# expand() math
# ---------------------------------------------------------------------------


def _basic_config(
    cells: list[SweepCell] | None = None,
    symbols: list[tuple[str, int]] | None = None,
) -> SweepConfig:
    start, end = _from_to()
    return SweepConfig(
        label="t",
        symbols=symbols
        or [("RELIANCE", 738561), ("INFY", 408065)],
        cells=cells
        or [
            SweepCell(
                strategy="ema_crossover",
                param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            )
        ],
        timeframe="5minute",
        from_date=start,
        to_date=end,
        qty=1,
    )


def test_expand_basic_cartesian() -> None:
    config = _basic_config()
    out = expand(config)
    # 2 symbols × (2×2) = 8.
    assert len(out) == 8
    symbols_in_out = {s for s, _, _, _ in out}
    assert symbols_in_out == {"RELIANCE", "INFY"}
    strategies_in_out = {strat for _, _, strat, _ in out}
    assert strategies_in_out == {"ema_crossover"}
    # Each combination of fast/slow appears for each symbol.
    grids = {
        (sym, p["fast_period"], p["slow_period"])
        for sym, _, _, p in out
    }
    assert len(grids) == 8


def test_expand_single_value_grid() -> None:
    cell = SweepCell(strategy="ema_crossover", param_grid={"atr_period": [14]})
    out = expand(_basic_config(cells=[cell]))
    assert len(out) == 2  # 2 symbols × 1 combo
    for _, _, _, params in out:
        assert params == {"atr_period": 14}


def test_expand_empty_grid_uses_defaults() -> None:
    cell = SweepCell(strategy="ema_crossover", param_grid={})
    out = expand(_basic_config(cells=[cell]))
    assert len(out) == 2  # 2 symbols × 1 default combo
    for _, _, _, params in out:
        assert params == {}


def test_expand_unknown_param_rejected() -> None:
    cell = SweepCell(
        strategy="ema_crossover", param_grid={"not_a_param": [1, 2]}
    )
    with pytest.raises(ValueError, match="unknown param"):
        expand(_basic_config(cells=[cell]))


def test_expand_unknown_strategy_rejected() -> None:
    cell = SweepCell(strategy="ghost_strategy", param_grid={})
    with pytest.raises(ValueError, match="unknown strategy"):
        expand(_basic_config(cells=[cell]))


def test_expand_out_of_bounds_rejected() -> None:
    cell = SweepCell(
        strategy="ema_crossover", param_grid={"fast_period": [-1]}
    )
    with pytest.raises(ValueError, match="outside"):
        expand(_basic_config(cells=[cell]))


def test_expand_int_param_rejects_non_integral() -> None:
    cell = SweepCell(
        strategy="ema_crossover", param_grid={"fast_period": [5.5]}
    )
    with pytest.raises(ValueError, match="integer"):
        expand(_basic_config(cells=[cell]))


def test_expand_empty_value_list_rejected() -> None:
    cell = SweepCell(
        strategy="ema_crossover", param_grid={"fast_period": []}
    )
    with pytest.raises(ValueError, match="non-empty"):
        expand(_basic_config(cells=[cell]))


def test_expand_cap_exceeded() -> None:
    # 3 symbols × 1 cell with 251 combos = 753 > 500.
    cell = SweepCell(
        strategy="ema_crossover",
        param_grid={"fast_period": list(range(1, 18))},  # 17
    )
    config = SweepConfig(
        label="t",
        symbols=[
            ("RELIANCE", 738561),
            ("INFY", 408065),
            ("RELIANCE", 738561),
        ],
        cells=[cell, cell, cell, cell, cell, cell, cell, cell, cell, cell],
        timeframe="5minute",
        from_date=_from_to()[0],
        to_date=_from_to()[1],
    )
    with pytest.raises(ValueError, match="too large"):
        expand(config)


# ---------------------------------------------------------------------------
# SweepRunner.create persists queued row
# ---------------------------------------------------------------------------


def test_create_persists_queued_row(tmp_path: Path) -> None:
    settings = AppSettings.model_validate({})
    conn = _build_dashboard_conn(tmp_path)
    try:
        instruments = _populate_instruments(conn, settings)
        runner = BacktestRunner(conn, settings=settings)
        sweep_runner = SweepRunner(
            conn,
            settings=settings,
            runner=runner,
            instruments=instruments,
            dashboard_db_path=tmp_path / "dash.duckdb",
        )
        sweep_id = sweep_runner.create(_basic_config())
        snapshot = sweep_runner.status(sweep_id)
        assert snapshot is not None
        assert snapshot.status == "queued"
        assert snapshot.total == 8
        assert snapshot.completed == 0
        assert snapshot.failed == 0
    finally:
        conn.close()


def test_create_rejects_unknown_symbol(tmp_path: Path) -> None:
    settings = AppSettings.model_validate({})
    conn = _build_dashboard_conn(tmp_path)
    try:
        instruments = _populate_instruments(conn, settings)
        runner = BacktestRunner(conn, settings=settings)
        sweep_runner = SweepRunner(
            conn,
            settings=settings,
            runner=runner,
            instruments=instruments,
            dashboard_db_path=tmp_path / "dash.duckdb",
        )
        config = _basic_config(symbols=[("GHOST", 999_999)])
        with pytest.raises(ValueError, match="unknown instrument"):
            sweep_runner.create(config)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SweepRunner.run end-to-end with synthetic bars
# ---------------------------------------------------------------------------


def _synthetic_loader(
    symbol: str,
    instrument_token: int,
    timeframe: str,
    from_date: datetime,
    to_date: datetime,
) -> BarsLoadResult:
    frame = make_synthetic_bars(200, seed=hash(symbol) % 1024)
    meta: dict[str, Any] = {
        "type": "synthetic",
        "instrument_token": int(instrument_token),
        "timeframe": timeframe,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "rows_fetched": 200,
        "rows_stored": 200,
        "gaps_filled": 0,
    }
    return BarsLoadResult(frame=frame, source_meta=meta)


def test_run_executes_all_cells_and_records_sweep_id(tmp_path: Path) -> None:
    settings = AppSettings.model_validate({})
    conn = _build_dashboard_conn(tmp_path)
    try:
        instruments = _populate_instruments(conn, settings)
        runner = BacktestRunner(conn, settings=settings)
        sweep_runner = SweepRunner(
            conn,
            settings=settings,
            runner=runner,
            instruments=instruments,
            dashboard_db_path=tmp_path / "dash.duckdb",
            bars_loader=_synthetic_loader,
        )
        config = _basic_config(
            cells=[
                SweepCell(
                    strategy="ema_crossover",
                    param_grid={"fast_period": [5, 10]},
                )
            ]
        )
        sweep_id = sweep_runner.create(config)
        asyncio.run(sweep_runner.run(sweep_id))

        snapshot = sweep_runner.status(sweep_id)
        assert snapshot is not None
        assert snapshot.status == "done"
        assert snapshot.completed + snapshot.failed == snapshot.total
        # Persisted runs should all carry the sweep_id.
        rows = conn.execute(
            "SELECT COUNT(*) FROM backtest_runs WHERE sweep_id = ?",
            [sweep_id],
        ).fetchone()
        assert rows is not None and int(rows[0]) == snapshot.completed

        leaderboard = sweep_runner.leaderboard(sweep_id)
        assert len(leaderboard) == snapshot.completed
        # Sorted DESC by total_pnl.
        pnls = [row.total_pnl for row in leaderboard]
        assert pnls == sorted(pnls, reverse=True)

        heatmap = sweep_runner.heatmap(sweep_id)
        assert heatmap["symbols"] == ["RELIANCE", "INFY"]
        assert heatmap["strategies"] == ["ema_crossover"]
        assert len(heatmap["cells"]) == 2
        assert len(heatmap["cells"][0]) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API smoke + page renders
# ---------------------------------------------------------------------------


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.5\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n", encoding="utf-8")
    settings = AppSettings.model_validate({}).model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "SWEEP_TEST_KILL",
            "state_db_path": tmp_path / "orders.duckdb",
        }
    )
    state = AppState(
        settings=settings,
        config_path=config_path,
        env_path=env_path,
        dashboard_db_path=tmp_path / "dash.duckdb",
        journal_path=None,
    )
    instruments = state.instruments()
    instruments._fetcher = _instruments_fetcher  # noqa: SLF001
    instruments.refresh()
    set_app_state(state)
    try:
        yield state
    finally:
        set_app_state(None)
        state.close()


@pytest.fixture
def client(app_state: AppState) -> Iterator[TestClient]:
    app = create_app(app_state, dev=True)

    # Force the sweep runner to use synthetic bars so the API round-trip
    # never reaches Kite. We replace the helper in both route modules
    # that imported it before TestClient starts driving traffic.
    import dashboard.routes.api as api_module
    import dashboard.routes.backtests as backtests_module
    from dashboard.routes._common import get_sweep_runner as _original_factory

    def _patched(state: AppState) -> SweepRunner:
        runner = _original_factory(state)
        runner._bars_loader = _synthetic_loader  # noqa: SLF001
        return runner

    originals: dict[Any, Any] = {}
    for module in (api_module, backtests_module):
        originals[module] = getattr(module, "get_sweep_runner")  # noqa: B009
        setattr(module, "get_sweep_runner", _patched)  # noqa: B010

    try:
        with TestClient(app) as c:
            yield c
    finally:
        for module, original_get in originals.items():
            setattr(module, "get_sweep_runner", original_get)  # noqa: B010


def _sweep_payload(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    start, end = _from_to()
    body: dict[str, Any] = {
        "label": "smoke",
        "symbols": [
            {"tradingsymbol": "RELIANCE", "instrument_token": 738561},
        ],
        "cells": [
            {
                "strategy": "ema_crossover",
                "param_grid": {"fast_period": [5, 10]},
            }
        ],
        "timeframe": "5minute",
        "from_date": start.isoformat(),
        "to_date": end.isoformat(),
        "qty": 1,
    }
    if extra:
        body.update(extra)
    return body


def _poll_until_done(
    client: TestClient, sweep_id: str, *, max_attempts: int = 60
) -> dict[str, Any]:
    """Poll the status endpoint until the sweep terminates or fails.

    The sweep task lives on the TestClient's portal event loop; we
    cannot ``await`` it from this thread. Polling drives the loop
    indirectly by issuing fresh requests through ``TestClient``.
    """
    for _ in range(max_attempts):
        resp = client.get(f"/api/backtest/sweep/{sweep_id}/status")
        assert resp.status_code == 200, resp.text
        payload: dict[str, Any] = resp.json()
        if payload["status"] in {"done", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"sweep {sweep_id} did not terminate in time")


def test_api_sweep_new_returns_id(client: TestClient, app_state: AppState) -> None:
    response = client.post("/api/backtest/sweep/new", json=_sweep_payload())
    assert response.status_code == 200, response.text
    sweep_id = response.json()["id"]
    assert isinstance(sweep_id, str) and sweep_id

    body = _poll_until_done(client, sweep_id)
    assert body["status"] == "done"

    leaderboard_resp = client.get(
        f"/api/backtest/sweep/{sweep_id}/leaderboard"
    )
    assert leaderboard_resp.status_code == 200
    rows = leaderboard_resp.json()["rows"]
    assert len(rows) == body["completed"]


def test_api_sweep_new_rejects_unknown_strategy(client: TestClient) -> None:
    payload = _sweep_payload(
        {
            "cells": [
                {"strategy": "ghost", "param_grid": {}},
            ],
        }
    )
    response = client.post("/api/backtest/sweep/new", json=payload)
    assert response.status_code == 400
    assert "unknown strategy" in response.json()["detail"]


def test_api_sweep_new_rejects_unknown_param(client: TestClient) -> None:
    payload = _sweep_payload(
        {
            "cells": [
                {
                    "strategy": "ema_crossover",
                    "param_grid": {"not_a_param": [1, 2]},
                }
            ],
        }
    )
    response = client.post("/api/backtest/sweep/new", json=payload)
    assert response.status_code == 400
    assert "unknown param" in response.json()["detail"]


def test_api_sweep_new_rejects_cap_exceeded(client: TestClient) -> None:
    # ema_crossover declares fast_period and slow_period in [1, 200];
    # 16 × 16 × 2 symbols = 512 > MAX_SWEEP_CELLS (500).
    big_grid = {
        "fast_period": list(range(1, 17)),
        "slow_period": list(range(2, 18)),
    }
    payload = _sweep_payload(
        {
            "cells": [
                {"strategy": "ema_crossover", "param_grid": big_grid},
            ],
            "symbols": [
                {"tradingsymbol": "RELIANCE", "instrument_token": 738561},
                {"tradingsymbol": "INFY", "instrument_token": 408065},
            ],
        }
    )
    response = client.post("/api/backtest/sweep/new", json=payload)
    assert response.status_code == 400
    assert (
        "too large" in response.json()["detail"]
        or str(MAX_SWEEP_CELLS) in response.json()["detail"]
    )


def test_api_sweep_new_rejects_unknown_symbol(client: TestClient) -> None:
    payload = _sweep_payload(
        {
            "symbols": [
                {"tradingsymbol": "GHOST", "instrument_token": 99999},
            ],
        }
    )
    response = client.post("/api/backtest/sweep/new", json=payload)
    assert response.status_code == 400
    assert "unknown instrument" in response.json()["detail"]


def test_sweep_pages_render(client: TestClient, app_state: AppState) -> None:
    new_resp = client.get("/backtests/sweep/new")
    assert new_resp.status_code == 200
    assert "sweep" in new_resp.text.lower()

    create_resp = client.post("/api/backtest/sweep/new", json=_sweep_payload())
    sweep_id = create_resp.json()["id"]
    _poll_until_done(client, sweep_id)
    detail_resp = client.get(f"/backtests/sweep/{sweep_id}")
    assert detail_resp.status_code == 200
