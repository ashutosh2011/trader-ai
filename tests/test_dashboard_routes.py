"""HTTP-level tests for dashboard page routes — each GET returns 200."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.state import AppState
from execution.order_state import OrderRecord, OrderState


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    """Build a fresh :class:`AppState` rooted in ``tmp_path``."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "risk:\n  max_loss_per_trade_pct: 0.5\n  daily_loss_cap_pct: 2.0\n",
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\nKITE_API_KEY=test\nKITE_API_SECRET=secret\n")
    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text("")
    settings = AppSettings.model_validate(
        {
            "risk": {"max_loss_per_trade_pct": 0.5, "daily_loss_cap_pct": 2.0},
        }
    )
    settings = settings.model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "DASHBOARD_TEST_KILL",
            "state_db_path": tmp_path / "orders.duckdb",
        }
    )

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


@pytest.fixture
def client(app_state: AppState) -> Iterator[TestClient]:
    """Build a TestClient over the fully-wired dashboard app."""
    app = create_app(app_state, dev=True)
    with TestClient(app) as c:
        yield c


def test_overview_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "Overview" in body
    assert "kill switch" in body


def test_live_page(client: TestClient) -> None:
    response = client.get("/live")
    assert response.status_code == 200
    assert "Live" in response.text
    assert "open positions" in response.text


def test_orders_page_empty(client: TestClient) -> None:
    response = client.get("/orders")
    assert response.status_code == 200
    assert "Orders" in response.text
    assert "no records match the filter." in response.text


def test_journal_page(client: TestClient) -> None:
    response = client.get("/journal")
    assert response.status_code == 200
    assert "Journal" in response.text


def test_backtests_page(client: TestClient) -> None:
    response = client.get("/backtests")
    assert response.status_code == 200
    body = response.text
    assert "Backtests" in body
    assert "run new" in body
    # Prominent data-source toggle replaced the cluttered dropdown — both
    # buttons must render; the hidden input is the source of truth.
    assert 'id="bt-source-toggle"' in body
    assert 'data-value="synthetic"' in body
    assert 'data-value="kite"' in body
    assert 'id="bt-source"' in body
    # Conditional field blocks + error panel both present.
    assert 'id="bt-synth-fields"' in body
    assert 'id="bt-kite-fields"' in body
    assert 'id="bt-error"' in body
    # Past-runs table shows a source column so synthetic vs kite is obvious.
    assert ">source<" in body


def test_config_page(client: TestClient) -> None:
    response = client.get("/config")
    assert response.status_code == 200
    assert "Config" in response.text
    assert "config-text" in response.text


def test_kite_page(client: TestClient) -> None:
    response = client.get("/kite")
    assert response.status_code == 200
    assert "Kite" in response.text


def test_strategies_page(client: TestClient) -> None:
    response = client.get("/strategies")
    assert response.status_code == 200
    assert "Strategies" in response.text
    assert "ema_crossover" in response.text


def test_api_overview_state(client: TestClient) -> None:
    response = client.get("/api/overview/state")
    assert response.status_code == 200
    data = response.json()
    assert "kill_active" in data
    assert "open_positions" in data
    assert data["kill_active"] is False


def test_api_orders(client: TestClient, app_state: AppState) -> None:
    now = datetime.now().astimezone()
    store = app_state.order_store()
    store.upsert(
        OrderRecord(
            client_order_id="x1",
            symbol="FOO",
            side="BUY",
            qty=1,
            entry_price=100.0,
            stop_loss=95.0,
            target=110.0,
            state=OrderState.ENTERED,
            signal_ts=now,
            created_at=now,
            updated_at=now,
            strategy_id="ema_crossover",
        )
    )
    response = client.get("/api/orders?state=ENTERED")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["rows"][0]["symbol"] == "FOO"


def test_api_journal_tail(client: TestClient, app_state: AppState) -> None:
    assert app_state.journal_path is not None
    app_state.journal_path.write_text(
        json.dumps({"ts": "2026-05-21T10:00:00+05:30", "event": "signal", "symbol": "FOO"})
        + "\n"
    )
    response = client.get("/api/journal/tail")
    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is True
    assert len(data["entries"]) == 1
    assert data["entries"][0]["event"] == "signal"


def test_partial_overview_state(client: TestClient) -> None:
    response = client.get("/_partials/overview/state")
    assert response.status_code == 200
    assert "today realized pnl" in response.text


def test_partial_live_state(client: TestClient) -> None:
    response = client.get("/_partials/live/state")
    assert response.status_code == 200
    assert "open positions" in response.text


def test_partial_journal_tail(client: TestClient) -> None:
    response = client.get("/_partials/journal/tail")
    assert response.status_code == 200


def test_backtest_detail_404(client: TestClient) -> None:
    response = client.get("/backtests/does-not-exist")
    assert response.status_code == 404
