"""Action-level tests for dashboard write endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from kiteconnect.exceptions import TokenException

from config.settings import AppSettings, KiteConfig
from dashboard.server import create_app
from dashboard.state import AppState
from execution.order_state import OrderRecord, OrderState


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "risk:\n  max_loss_per_trade_pct: 0.5\n  daily_loss_cap_pct: 2.0\n",
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\nKITE_API_KEY=test\nKITE_API_SECRET=secret\n")
    settings = AppSettings.model_validate({})
    settings = settings.model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "DASHBOARD_TEST_KILL",
            "state_db_path": tmp_path / "orders.duckdb",
            "kite": KiteConfig(api_key="test", api_secret="secret"),
        }
    )
    state = AppState(
        settings=settings,
        config_path=config_path,
        env_path=env_path,
        dashboard_db_path=tmp_path / "dash.duckdb",
        journal_path=tmp_path / "journal.jsonl",
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


# ---------------------------------------------------------------------------
# kill switch toggle
# ---------------------------------------------------------------------------


def test_kill_toggle_round_trip(client: TestClient, app_state: AppState) -> None:
    response = client.post("/api/kill/toggle", json={"enabled": True})
    assert response.status_code == 200
    assert response.json()["active"] is True
    assert app_state.settings.kill_switch_file.is_file()

    response = client.post("/api/kill/toggle", json={"enabled": False})
    assert response.status_code == 200
    assert response.json()["active"] is False
    assert not app_state.settings.kill_switch_file.is_file()


# ---------------------------------------------------------------------------
# config save / validate
# ---------------------------------------------------------------------------


_VALID_YAML = """\
risk:
  max_loss_per_trade_pct: 0.4
  daily_loss_cap_pct: 1.5
"""


def test_config_validate_ok(client: TestClient) -> None:
    response = client.post("/api/config/validate", json={"yaml": _VALID_YAML})
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_config_validate_invalid(client: TestClient) -> None:
    response = client.post(
        "/api/config/validate",
        json={"yaml": "risk:\n  max_loss_per_trade_pct: -1.0\n"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert data["issues"]


def test_config_save_writes_and_reloads(client: TestClient, app_state: AppState) -> None:
    response = client.post("/api/config/save", json={"yaml": _VALID_YAML})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    # backup_path should be set because there was a prior file
    assert payload["backup"] is not None
    assert Path(payload["backup"]).is_file()
    new_text = app_state.config_path.read_text(encoding="utf-8")
    assert "max_loss_per_trade_pct: 0.4" in new_text
    # AppState reloaded the settings.
    assert app_state.settings.risk.max_loss_per_trade_pct == 0.4


def test_config_save_invalid_no_write(client: TestClient, app_state: AppState) -> None:
    prior = app_state.config_path.read_text(encoding="utf-8")
    response = client.post(
        "/api/config/save",
        json={"yaml": "risk:\n  max_loss_per_trade_pct: -5\n"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert app_state.config_path.read_text(encoding="utf-8") == prior


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


def test_flatten_requires_confirm_token(client: TestClient) -> None:
    response = client.post("/api/flatten", json={"confirm": "no"})
    assert response.status_code == 422  # pydantic validation


def test_flatten_without_kite_returns_400(tmp_path: Path) -> None:
    settings = AppSettings.model_validate({}).model_copy(
        update={
            "kite": KiteConfig(),
            "kill_switch_file": tmp_path / "KILL",
            "state_db_path": tmp_path / "orders.duckdb",
        }
    )
    state = AppState(
        settings=settings,
        config_path=tmp_path / "config.yaml",
        env_path=tmp_path / ".env",
        dashboard_db_path=tmp_path / "dash.duckdb",
        journal_path=None,
    )
    app = create_app(state, dev=True)
    try:
        with TestClient(app) as c:
            response = c.post("/api/flatten", json={"confirm": "FLATTEN"})
        assert response.status_code == 400
        assert "no_kite_broker_configured" in response.json()["detail"]
    finally:
        state.close()


def test_flatten_calls_kite_broker(
    app_state: AppState,
) -> None:
    new_settings = app_state.settings.model_copy(
        update={
            "kite": app_state.settings.kite.model_copy(
                update={"access_token": "tkn", "api_key": "test", "api_secret": "secret"}
            )
        }
    )
    app_state._settings = new_settings  # noqa: SLF001 (test override of private)
    assert app_state.settings.kite_configured() is True

    flatten_called: list[bool] = []

    def fake_flatten(self: Any) -> None:  # noqa: ARG001
        flatten_called.append(True)

    app = create_app(app_state, dev=True)
    with (
        patch("execution.kite.KiteBroker.flatten_all", new=fake_flatten),
        patch("data.kite_client.KiteClient.from_settings", return_value=object()),
        TestClient(app) as c,
    ):
        response = c.post("/api/flatten", json={"confirm": "FLATTEN"})
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert flatten_called == [True]


# ---------------------------------------------------------------------------
# kite exchange
# ---------------------------------------------------------------------------


def test_kite_exchange_writes_env(
    client: TestClient,
    app_state: AppState,
) -> None:
    with patch(
        "dashboard.services.kite_auth.exchange_request_token", return_value="ACCESS_TOKEN_123"
    ):
        response = client.post("/api/kite/exchange", json={"request_token": "rt"})
    assert response.status_code == 200
    text = app_state.env_path.read_text(encoding="utf-8")
    assert "KITE_ACCESS_TOKEN=ACCESS_TOKEN_123" in text
    # Other lines preserved.
    assert "KITE_API_KEY=test" in text
    assert "KITE_API_SECRET=secret" in text


def test_kite_exchange_rejects_empty_token(client: TestClient) -> None:
    response = client.post("/api/kite/exchange", json={"request_token": ""})
    assert response.status_code == 400


def test_kite_exchange_invalid_checksum_returns_400(client: TestClient) -> None:
    with patch(
        "dashboard.services.kite_auth.exchange_request_token",
        side_effect=TokenException("Invalid `checksum`."),
    ):
        response = client.post("/api/kite/exchange", json={"request_token": "bad"})
    assert response.status_code == 400
    assert "Kite rejected the request token" in response.json()["detail"]


# ---------------------------------------------------------------------------
# orders mark
# ---------------------------------------------------------------------------


def _add_open_order(app_state: AppState, *, client_order_id: str = "x1") -> None:
    now = datetime.now().astimezone()
    app_state.order_store().upsert(
        OrderRecord(
            client_order_id=client_order_id,
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


def test_order_mark_failed(client: TestClient, app_state: AppState) -> None:
    _add_open_order(app_state)
    response = client.post(
        "/api/orders/x1/mark",
        json={"state": "FAILED", "reason": "test"},
    )
    assert response.status_code == 200
    record = app_state.order_store().get("x1")
    assert record is not None
    assert record.state == OrderState.FAILED


def test_order_mark_not_found(client: TestClient) -> None:
    response = client.post(
        "/api/orders/missing/mark",
        json={"state": "CANCELLED"},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# strategies toggle
# ---------------------------------------------------------------------------


def test_strategies_toggle(client: TestClient) -> None:
    response = client.post("/api/strategies/ema_crossover/toggle")
    assert response.status_code == 200
    data = response.json()
    assert data["strategy_id"] == "ema_crossover"
    # Default is enabled=True, so toggling flips to False.
    assert data["enabled"] is False

    response = client.post("/api/strategies/ema_crossover/toggle")
    assert response.json()["enabled"] is True


# ---------------------------------------------------------------------------
# backtest run end-to-end
# ---------------------------------------------------------------------------


def test_backtest_run_end_to_end(client: TestClient) -> None:
    payload = {
        "strategy": "ema_crossover",
        "symbol": "SYNTH",
        "bars_count": 200,
        "qty": 1,
        "seed": 7,
        "params": {"fast_period": 5, "slow_period": 12, "atr_period": 7},
    }
    response = client.post("/api/backtest/run", json=payload)
    assert response.status_code == 200
    run_id = response.json()["id"]

    response = client.get(f"/backtests/{run_id}")
    assert response.status_code == 200
    body = response.text
    assert "Backtest" in body
    assert "metrics" in body


def test_backtest_run_unknown_strategy(client: TestClient) -> None:
    response = client.post(
        "/api/backtest/run",
        json={"strategy": "nope", "symbol": "X", "bars_count": 50},
    )
    assert response.status_code == 400


def test_backtest_run_kite_token_error_returns_400(client: TestClient) -> None:
    class FailingRunner:
        def run(self, **kwargs: object) -> str:
            raise TokenException("Incorrect `api_key` or `access_token`.")

    with patch("dashboard.routes.api.get_backtest_runner", return_value=FailingRunner()):
        response = client.post(
            "/api/backtest/run",
            json={
                "strategy": "ema_crossover",
                "symbol": "INFY",
                "bars_count": 500,
                "data_source": "kite",
                "instrument_token": 408065,
                "timeframe": "5minute",
                "from_date": "2026-05-20T09:15",
                "to_date": "2026-05-20T15:30",
            },
        )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Kite rejected the historical-data request" in detail
    assert "Refresh today's access token" in detail
