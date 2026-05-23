"""Tests for the Kite-instruments cache + /api/instruments endpoints."""

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
from dashboard.state import INSTRUMENTS_META_SCHEMA, INSTRUMENTS_SCHEMA, AppState


def _full_fetcher(_settings: AppSettings, _exchange: str) -> list[dict[str, Any]]:
    return list(_FAKE_KITE_ROWS)


def _single_fetcher(_settings: AppSettings, _exchange: str) -> list[dict[str, Any]]:
    return [_FAKE_KITE_ROWS[0]]


_FAKE_KITE_ROWS: list[dict[str, Any]] = [
    {
        "instrument_token": 738561,
        "exchange_token": 2885,
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
        "exchange_token": 1594,
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
    {
        "instrument_token": 5633,
        "exchange_token": 22,
        "tradingsymbol": "ACC",
        "name": "ACC Limited",
        "last_price": 2200.0,
        "expiry": "",
        "strike": 0,
        "tick_size": 0.05,
        "lot_size": 1,
        "instrument_type": "EQ",
        "segment": "NSE",
        "exchange": "NSE",
    },
    # Filtered out by the EQ-only refresh.
    {
        "instrument_token": 1,
        "exchange_token": 0,
        "tradingsymbol": "NIFTYFUT",
        "name": "Nifty Future",
        "last_price": 0.0,
        "expiry": "2026-05-29",
        "strike": 0,
        "tick_size": 0.05,
        "lot_size": 25,
        "instrument_type": "FUT",
        "segment": "NFO-FUT",
        "exchange": "NFO",
    },
]


def _build_service(
    tmp_path: Path,
) -> tuple[duckdb.DuckDBPyConnection, InstrumentsService]:
    conn = duckdb.connect(str(tmp_path / "instr.duckdb"))
    conn.execute(INSTRUMENTS_SCHEMA)
    conn.execute(INSTRUMENTS_META_SCHEMA)
    settings = AppSettings.model_validate({})
    service = InstrumentsService(conn, settings=settings, fetcher=_full_fetcher)
    service.ensure_schema()
    return conn, service


def test_refresh_writes_rows_and_filters_non_eq(tmp_path: Path) -> None:
    conn, service = _build_service(tmp_path)
    try:
        count = service.refresh()
        assert count == 3  # FUT row filtered out
        snapshot = service.status()
        assert snapshot["row_count"] == 3
        assert snapshot["last_refresh"] is not None
        assert snapshot["stale"] is False
    finally:
        conn.close()


def test_search_prefix_then_substring_case_insensitive(tmp_path: Path) -> None:
    conn, service = _build_service(tmp_path)
    try:
        service.refresh()
        results = service.search("rel")
        assert results[0].tradingsymbol == "RELIANCE"
        # substring on name should still surface ACC when 'limited' is queried.
        results = service.search("limited")
        assert any(r.tradingsymbol == "ACC" for r in results)
    finally:
        conn.close()


def test_search_empty_returns_alphabetical(tmp_path: Path) -> None:
    conn, service = _build_service(tmp_path)
    try:
        service.refresh()
        results = service.search("", limit=10)
        assert [r.tradingsymbol for r in results] == ["ACC", "INFY", "RELIANCE"]
    finally:
        conn.close()


def test_get_by_symbol_and_token(tmp_path: Path) -> None:
    conn, service = _build_service(tmp_path)
    try:
        service.refresh()
        match = service.get_by_symbol("infy")
        assert match is not None and match.instrument_token == 408065
        match_token = service.get_by_token(738561)
        assert match_token is not None and match_token.tradingsymbol == "RELIANCE"
        assert service.get_by_symbol("nope") is None
        assert service.get_by_token(9999) is None
    finally:
        conn.close()


def test_refresh_replaces_rows(tmp_path: Path) -> None:
    conn, service = _build_service(tmp_path)
    try:
        service.refresh()
        # Now swap the fetcher for a smaller dump and refresh again.
        service._fetcher = _single_fetcher  # noqa: SLF001
        count = service.refresh()
        assert count == 1
        assert service.get_by_symbol("INFY") is None
        assert service.get_by_symbol("RELIANCE") is not None
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
            "kill_switch_env": "INSTR_TEST_KILL",
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
    instruments._fetcher = _full_fetcher  # noqa: SLF001
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


def test_api_instruments_search_matches_legacy_shape(client: TestClient) -> None:
    response = client.get("/api/instruments/search?q=infy&limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert "results" in payload and isinstance(payload["results"], list)
    first = payload["results"][0]
    # Must include the rich Instrument fields used by the new sweep UI.
    for key in (
        "instrument_token",
        "tradingsymbol",
        "name",
        "exchange",
        "instrument_type",
        "segment",
        "tick_size",
        "lot_size",
        "last_price",
    ):
        assert key in first


def test_api_symbols_search_uses_instruments_table(client: TestClient) -> None:
    """The legacy /api/symbols/search endpoint now reads from instruments."""
    response = client.get("/api/symbols/search?q=infy&limit=5")
    assert response.status_code == 200
    payload = response.json()
    first = payload["results"][0]
    assert first["symbol"] == "INFY"
    assert first["instrument_token"] == 408065


def test_api_instruments_status_reports_count(client: TestClient) -> None:
    response = client.get("/api/instruments/status")
    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 3
    assert "last_refresh" in body
    assert "stale" in body
