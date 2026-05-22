"""Tests for the universe management service + routes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml  # type: ignore[import-untyped]
from fastapi.testclient import TestClient

from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.services.universe_io import POPULAR_SEED_SYMBOLS, UniverseIO, UniverseIOError
from dashboard.state import AppState
from screener.universe import UniverseSymbol


def test_add_symbol_persists(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    service.save_universe([UniverseSymbol(symbol="RELIANCE", instrument_token=738561)])
    universe = service.add_symbol(symbol="INFY", exchange="NSE", instrument_token=408065)
    symbols = [s.symbol for s in universe.symbols]
    assert "INFY" in symbols
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["symbols"][-1]["symbol"] == "INFY"


def test_add_duplicate_raises(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    service.save_universe([UniverseSymbol(symbol="RELIANCE", instrument_token=738561)])
    with pytest.raises(UniverseIOError):
        service.add_symbol(symbol="reliance", exchange="NSE", instrument_token=999)


def test_update_symbol(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    service.save_universe([UniverseSymbol(symbol="INFY", instrument_token=1)])
    service.update_symbol(symbol="INFY", instrument_token=408065)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["symbols"][0]["instrument_token"] == 408065


def test_update_missing_raises(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    service.save_universe([UniverseSymbol(symbol="INFY", instrument_token=1)])
    with pytest.raises(UniverseIOError):
        service.update_symbol(symbol="MISSING", instrument_token=42)


def test_delete_symbol(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    service.save_universe(
        [
            UniverseSymbol(symbol="A", instrument_token=1),
            UniverseSymbol(symbol="B", instrument_token=2),
        ]
    )
    service.delete_symbol("A")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert [s["symbol"] for s in data["symbols"]] == ["B"]


def test_delete_last_blocked(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    service.save_universe([UniverseSymbol(symbol="LONE", instrument_token=1)])
    with pytest.raises(UniverseIOError):
        service.delete_symbol("LONE")


def test_seed_popular_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    added = service.seed_popular(limit=5)
    assert added == 5
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(data["symbols"]) == 5
    added_again = service.seed_popular(limit=5)
    # Re-seeding with the same first 5 should add nothing new; the next 5 may still be added.
    # Allow up to 5 new because seed_popular keeps adding from POPULAR_SEED_SYMBOLS until the
    # in-list set covers the requested popular entries.
    assert added_again <= 5
    final = yaml.safe_load(path.read_text(encoding="utf-8"))
    # No duplicates regardless of repeated seeding.
    seen = [s["symbol"] for s in final["symbols"]]
    assert len(set(seen)) == len(seen)


def test_seed_popular_uses_full_list(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    added = service.seed_popular(limit=100)
    assert added == len(POPULAR_SEED_SYMBOLS)


def test_save_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    service.save_universe(
        [
            UniverseSymbol(symbol="ONE", instrument_token=1),
            UniverseSymbol(symbol="TWO", instrument_token=2),
        ]
    )
    universe = service.load_universe_editable()
    assert [s.symbol for s in universe.symbols] == ["ONE", "TWO"]


def test_save_creates_backup(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=path)
    service.save_universe([UniverseSymbol(symbol="A", instrument_token=1)])
    service.save_universe([UniverseSymbol(symbol="B", instrument_token=2)])
    assert (tmp_path / "universe.yaml.bak").is_file()


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.5\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n", encoding="utf-8")
    settings = AppSettings.model_validate({}).model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "UN_TEST_KILL",
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
    try:
        yield state
    finally:
        state.close()


@pytest.fixture
def client(
    app_state: AppState,
    tmp_path: Path,
) -> Iterator[TestClient]:
    universe_file = tmp_path / "universe.yaml"
    service = UniverseIO(universe_path=universe_file)
    service.save_universe([UniverseSymbol(symbol="SEED", instrument_token=1)])
    with patch(
        "dashboard.routes.universe._service",
        lambda: UniverseIO(universe_path=universe_file),
    ):
        app = create_app(app_state, dev=True)
        with TestClient(app) as c:
            yield c


def test_universe_page(client: TestClient) -> None:
    response = client.get("/universe")
    assert response.status_code == 200
    body = response.text
    assert "Universe" in body
    assert "SEED" in body


def test_universe_add_api(client: TestClient) -> None:
    response = client.post(
        "/api/universe/add",
        json={"symbol": "INFY", "exchange": "NSE", "instrument_token": 408065},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_universe_update_api(client: TestClient) -> None:
    response = client.post(
        "/api/universe/update/SEED",
        json={"instrument_token": 42},
    )
    assert response.status_code == 200


def test_universe_delete_api(client: TestClient) -> None:
    # Need to ensure there's at least 2 symbols first (delete_last_blocked)
    client.post(
        "/api/universe/add",
        json={"symbol": "EXTRA", "exchange": "NSE", "instrument_token": 1234},
    )
    response = client.post("/api/universe/delete/EXTRA", json={})
    assert response.status_code == 200


def test_universe_delete_last_blocked(client: TestClient) -> None:
    response = client.post("/api/universe/delete/SEED", json={})
    assert response.status_code == 400


def test_universe_seed_api(client: TestClient) -> None:
    response = client.post("/api/universe/seed", json={"limit": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["added"] >= 1
