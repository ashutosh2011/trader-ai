"""Smoke tests for the new dashboard pages + nav grouping."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.state import AppState


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.5\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n", encoding="utf-8")
    settings = AppSettings.model_validate({}).model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "UX_TEST_KILL",
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


def test_nav_includes_new_links(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    for href in ("/llm", "/universe", "/reports"):
        assert href in body
    # Group titles render via title attribute.
    assert 'title="Run"' in body
    assert 'title="Strategy"' in body
    assert 'title="Setup"' in body
    assert 'title="Inspect"' in body


def test_static_toast_served(client: TestClient) -> None:
    response = client.get("/static/toast.js")
    assert response.status_code == 200
    assert "tbToast" in response.text


def test_static_symbol_picker_served(client: TestClient) -> None:
    response = client.get("/static/symbol-picker.js")
    assert response.status_code == 200
    assert "data-symbol-picker" in response.text


def test_backtests_page_uses_symbol_picker(client: TestClient) -> None:
    response = client.get("/backtests")
    assert response.status_code == 200
    body = response.text
    assert 'data-symbol-picker="true"' in body
    assert 'id="bt-symbol"' in body
    assert 'id="bt-symbol-dropdown"' in body


def test_inr_amount_filter() -> None:
    from dashboard.server import _inr_amount

    assert _inr_amount(1234.5) == "₹ 1,234.50"
    assert _inr_amount(-567.89) == "-₹ 567.89"
    assert _inr_amount(None) == "₹ —"
    assert _inr_amount(0) == "₹ 0.00"
