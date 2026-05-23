"""Tests for the symbol lookup service + /api/symbols endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest
from fastapi.testclient import TestClient

from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.services.instruments import InstrumentsService
from dashboard.services.symbol_lookup import SymbolLookupService
from dashboard.state import INSTRUMENTS_META_SCHEMA, INSTRUMENTS_SCHEMA, AppState


def _stub_fetcher(_settings: AppSettings, _exchange: str) -> list[dict[str, Any]]:
    return list(_INSTRUMENT_FIXTURES)


def _empty_fetcher(_settings: AppSettings, _exchange: str) -> list[dict[str, Any]]:
    return []


_INSTRUMENT_FIXTURES: list[dict[str, Any]] = [
    {
        "instrument_token": 738561,
        "tradingsymbol": "RELIANCE",
        "name": "Reliance Industries",
        "exchange": "NSE",
        "instrument_type": "EQ",
        "segment": "NSE",
        "tick_size": 0.05,
        "lot_size": 1,
        "last_price": 2500.0,
    },
    {
        "instrument_token": 408065,
        "tradingsymbol": "INFY",
        "name": "Infosys",
        "exchange": "NSE",
        "instrument_type": "EQ",
        "segment": "NSE",
        "tick_size": 0.05,
        "lot_size": 1,
        "last_price": 1500.0,
    },
    {
        "instrument_token": 2953217,
        "tradingsymbol": "TCS",
        "name": "Tata Consultancy Services",
        "exchange": "NSE",
        "instrument_type": "EQ",
        "segment": "NSE",
        "tick_size": 0.05,
        "lot_size": 1,
        "last_price": 3200.0,
    },
    {
        "instrument_token": 341249,
        "tradingsymbol": "HDFCBANK",
        "name": "HDFC Bank",
        "exchange": "NSE",
        "instrument_type": "EQ",
        "segment": "NSE",
        "tick_size": 0.05,
        "lot_size": 1,
        "last_price": 1700.0,
    },
]


def _make_instruments_service(
    tmp_path: Path,
) -> tuple[duckdb.DuckDBPyConnection, InstrumentsService]:
    conn = duckdb.connect(str(tmp_path / "lookup.duckdb"))
    conn.execute(INSTRUMENTS_SCHEMA)
    conn.execute(INSTRUMENTS_META_SCHEMA)
    settings = AppSettings.model_validate({})
    service = InstrumentsService(conn, settings=settings, fetcher=_stub_fetcher)
    service.ensure_schema()
    service.refresh()
    return conn, service


def test_list_symbols(tmp_path: Path) -> None:
    conn, instruments = _make_instruments_service(tmp_path)
    try:
        service = SymbolLookupService(instruments)
        entries = service.list_symbols(limit=50)
        symbols = sorted(e.symbol for e in entries)
        assert symbols == sorted(["RELIANCE", "INFY", "TCS", "HDFCBANK"])
    finally:
        conn.close()


def test_search_prefix_first(tmp_path: Path) -> None:
    conn, instruments = _make_instruments_service(tmp_path)
    try:
        service = SymbolLookupService(instruments)
        results = service.search("HD")
        assert results[0].symbol == "HDFCBANK"
    finally:
        conn.close()


def test_search_substring_match(tmp_path: Path) -> None:
    conn, instruments = _make_instruments_service(tmp_path)
    try:
        service = SymbolLookupService(instruments)
        results = service.search("BANK")
        assert any(r.symbol == "HDFCBANK" for r in results)
    finally:
        conn.close()


def test_search_exact_match_ranks_first(tmp_path: Path) -> None:
    conn, instruments = _make_instruments_service(tmp_path)
    try:
        service = SymbolLookupService(instruments)
        results = service.search("INFY")
        assert results[0].symbol == "INFY"
    finally:
        conn.close()


def test_search_empty_returns_full_list_up_to_limit(tmp_path: Path) -> None:
    conn, instruments = _make_instruments_service(tmp_path)
    try:
        service = SymbolLookupService(instruments)
        assert len(service.search("", limit=2)) == 2
        assert len(service.search("", limit=10)) == 4
    finally:
        conn.close()


def test_search_empty_table_returns_empty(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "empty.duckdb"))
    conn.execute(INSTRUMENTS_SCHEMA)
    conn.execute(INSTRUMENTS_META_SCHEMA)
    settings = AppSettings.model_validate({})
    instruments = InstrumentsService(conn, settings=settings, fetcher=_empty_fetcher)
    instruments.ensure_schema()
    try:
        service = SymbolLookupService(instruments)
        assert service.search("RELIANCE") == []
        assert service.list_symbols() == []
    finally:
        conn.close()


def test_find_symbol(tmp_path: Path) -> None:
    conn, instruments = _make_instruments_service(tmp_path)
    try:
        service = SymbolLookupService(instruments)
        entry = service.find_symbol("reliance")
        assert entry is not None
        assert entry.symbol == "RELIANCE"
        assert entry.instrument_token == 738561
    finally:
        conn.close()


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
    # Inject a fake fetcher into the lazy InstrumentsService and seed the
    # cache so the API endpoints below see populated rows.
    instruments = state.instruments()
    instruments._fetcher = _stub_fetcher  # noqa: SLF001
    instruments.refresh()
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
    response = client.get("/api/symbols/search?q=REL&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["results"], list)
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
