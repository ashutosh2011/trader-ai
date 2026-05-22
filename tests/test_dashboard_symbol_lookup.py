"""Tests for the symbol lookup service + /api/symbols endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]
from fastapi.testclient import TestClient

from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.services.symbol_lookup import SymbolLookupService
from dashboard.state import AppState


def _write_universe(path: Path) -> None:
    payload = {
        "symbols": [
            {"symbol": "RELIANCE", "instrument_token": 738561, "exchange": "NSE"},
            {"symbol": "INFY", "instrument_token": 408065, "exchange": "NSE"},
            {"symbol": "TCS", "instrument_token": 2953217, "exchange": "NSE"},
            {"symbol": "HDFCBANK", "instrument_token": 341249, "exchange": "NSE"},
        ]
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_list_symbols(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    _write_universe(path)
    service = SymbolLookupService(universe_path=path)
    entries = service.list_symbols()
    symbols = [e.symbol for e in entries]
    assert symbols == sorted(["RELIANCE", "INFY", "TCS", "HDFCBANK"])


def test_search_prefix_first(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    _write_universe(path)
    service = SymbolLookupService(universe_path=path)
    results = service.search("HD")
    assert results[0].symbol == "HDFCBANK"


def test_search_substring_match(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    _write_universe(path)
    service = SymbolLookupService(universe_path=path)
    results = service.search("BANK")
    assert any(r.symbol == "HDFCBANK" for r in results)


def test_search_exact_match_ranks_first(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    _write_universe(path)
    service = SymbolLookupService(universe_path=path)
    results = service.search("INFY")
    assert results[0].symbol == "INFY"


def test_search_empty_returns_full_list_up_to_limit(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    _write_universe(path)
    service = SymbolLookupService(universe_path=path)
    assert len(service.search("", limit=2)) == 2
    assert len(service.search("", limit=10)) == 4


def test_search_missing_universe_returns_empty(tmp_path: Path) -> None:
    service = SymbolLookupService(universe_path=tmp_path / "absent.yaml")
    assert service.search("RELIANCE") == []
    assert service.list_symbols() == []


def test_find_symbol(tmp_path: Path) -> None:
    path = tmp_path / "universe.yaml"
    _write_universe(path)
    service = SymbolLookupService(universe_path=path)
    entry = service.find_symbol("reliance")
    assert entry is not None
    assert entry.symbol == "RELIANCE"
    assert entry.instrument_token == 738561


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.5\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n", encoding="utf-8")
    settings = AppSettings.model_validate({}).model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "SL_TEST_KILL",
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
def client(app_state: AppState) -> Iterator[TestClient]:
    app = create_app(app_state, dev=True)
    with TestClient(app) as c:
        yield c


def test_api_symbols_search_returns_results(client: TestClient) -> None:
    # The bundled example file is the fallback when config/universe.yaml is absent
    # in the workspace; for the test harness we rely on the example being present.
    response = client.get("/api/symbols/search?q=REL&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["results"], list)
    # RELIANCE should be in the bundled example universe.
    assert any(r["symbol"] == "RELIANCE" for r in data["results"])


def test_api_symbols_get_404(client: TestClient) -> None:
    response = client.get("/api/symbols/NOPE_NOT_A_SYMBOL_XYZ")
    assert response.status_code == 404


def test_api_symbols_get_existing(client: TestClient) -> None:
    response = client.get("/api/symbols/RELIANCE")
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "RELIANCE"
    assert payload["exchange"] == "NSE"
