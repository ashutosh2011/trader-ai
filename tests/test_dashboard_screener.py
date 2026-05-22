"""HTTP tests for the screener dashboard routes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.settings import AppSettings, KiteConfig
from dashboard.server import create_app
from dashboard.state import AppState
from data.store import CandleStore


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.5\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n")
    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text("")

    settings = AppSettings.model_validate({})
    candle_path = tmp_path / "candles.duckdb"
    settings = settings.model_copy(
        update={
            # Wipe any Kite creds bleeding in from the project .env so
            # fetch_missing=True deterministically returns 400.
            "kite": KiteConfig(),
            "kite_api_key": None,
            "kite_access_token": None,
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "SCREENER_TEST_KILL",
            "state_db_path": tmp_path / "orders.duckdb",
            "data": settings.data.model_copy(update={"duckdb_path": candle_path}),
        }
    )

    # Pre-populate the candle store so the stub formula has something to scan.
    _seed_candles(candle_path, "RELIANCE")
    _seed_candles(candle_path, "INFY")

    state = AppState(
        settings=settings,
        config_path=config_path,
        env_path=env_path,
        dashboard_db_path=tmp_path / "dash.duckdb",
        journal_path=journal_path,
    )
    try:
        yield state
    finally:
        state.close()


def _seed_candles(path: Path, symbol: str) -> None:
    store = CandleStore(path)
    try:
        timestamps = pd.date_range(
            start="2024-01-01 09:15:00",
            periods=120,
            freq="1D",
            tz="Asia/Kolkata",
        )
        close = np.linspace(100.0, 200.0, 120)
        open_ = np.empty(120)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_,
                "high": np.maximum(open_, close) + 0.5,
                "low": np.minimum(open_, close) - 0.5,
                "close": close,
                "volume": np.full(120, 1_000.0),
            }
        )
        store.upsert_bars(symbol, "day", frame)
    finally:
        store.close()


@pytest.fixture
def client(app_state: AppState) -> Iterator[TestClient]:
    app = create_app(app_state, dev=True)
    with TestClient(app) as c:
        yield c


def test_screener_list_page(client: TestClient) -> None:
    response = client.get("/screener")
    assert response.status_code == 200
    body = response.text
    assert "Screener" in body
    assert "run screener" in body
    assert "id=\"sc-form\"" in body
    # Provider dropdown includes stub.
    assert ">stub<" in body


def test_screener_run_returns_run_id_and_persists(
    client: TestClient,
    app_state: AppState,
) -> None:
    response = client.post(
        "/api/screener/run",
        json={
            "provider": "stub",
            "fetch_missing": False,
            "bars_back": 200,
            "market_context_notes": "test run",
            "recent_index_summary": "flat",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "run_id" in payload
    assert payload["universe_size"] >= 1
    runs = app_state.screener_store().list_runs()
    assert any(r.id == payload["run_id"] for r in runs)


def test_screener_detail_page(client: TestClient) -> None:
    rid = client.post(
        "/api/screener/run",
        json={"provider": "stub", "fetch_missing": False, "bars_back": 200},
    ).json()["run_id"]
    response = client.get(f"/screener/{rid}")
    assert response.status_code == 200
    body = response.text
    assert rid in body
    assert "formula" in body
    assert "rationale" in body.lower()


def test_screener_detail_404(client: TestClient) -> None:
    response = client.get("/screener/does-not-exist")
    assert response.status_code == 404


def test_screener_run_rejects_bad_provider(client: TestClient) -> None:
    response = client.post(
        "/api/screener/run",
        json={"provider": "bogus", "fetch_missing": False},
    )
    assert response.status_code == 422


def test_screener_run_fetch_missing_without_kite_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/screener/run",
        json={"provider": "stub", "fetch_missing": True, "bars_back": 200},
    )
    assert response.status_code == 400
    assert "KITE" in response.json()["detail"] or "kite" in response.json()["detail"].lower()


def test_screener_nav_entry_present(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "/screener" in body
    assert "Screener" in body
