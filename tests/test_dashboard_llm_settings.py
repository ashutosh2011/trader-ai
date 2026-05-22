"""Tests for the dashboard LLM settings service + page routes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml  # type: ignore[import-untyped]
from fastapi.testclient import TestClient

from analyst.providers.mock import MockLLMProvider
from config.settings import AnalystProviderConfig, AppSettings
from dashboard.server import create_app
from dashboard.services.llm_settings import (
    LLMSettingsService,
    ProviderName,
)
from dashboard.state import AppState


def _settings_with_keys(
    *,
    anthropic: str | None = None,
    openai: str | None = None,
    google: str | None = None,
    default_provider: str = "mock",
) -> AppSettings:
    return AppSettings.model_validate(
        {
            "analyst": {
                "anthropic_api_key": anthropic,
                "openai_api_key": openai,
                "google_api_key": google,
                "default_provider": default_provider,
            }
        }
    )


def _reload_from_env(env: Path) -> AppSettings:
    """Tiny .env parser → AppSettings(analyst=...). Used as a test reload."""
    keys: dict[str, str] = {}
    if env.is_file():
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            keys[key.strip()] = value.strip().strip('"').strip("'")
    return AppSettings.model_validate(
        {
            "analyst": {
                "anthropic_api_key": keys.get("ANTHROPIC_API_KEY") or None,
                "openai_api_key": keys.get("OPENAI_API_KEY") or None,
                "google_api_key": keys.get("GOOGLE_API_KEY") or None,
            }
        }
    )


def _build_service(
    tmp_path: Path,
    *,
    settings: AppSettings | None = None,
) -> LLMSettingsService:
    env = tmp_path / ".env"
    cfg = tmp_path / "config.yaml"
    snap = settings if settings is not None else _settings_with_keys()
    return LLMSettingsService(
        settings=snap,
        env_path=env,
        config_path=cfg,
        provider_factory=lambda name, _config: MockLLMProvider("PONG", name=name),
        reload_settings=lambda: _reload_from_env(env),
    )


def test_read_status_masks_keys(tmp_path: Path) -> None:
    settings = _settings_with_keys(anthropic="sk-anth-1234567890abcd", openai=None)
    service = _build_service(tmp_path, settings=settings)
    snap = service.read_status()
    by_name = {p.name: p for p in snap.providers}
    assert by_name["anthropic"].configured is True
    assert by_name["anthropic"].preview.startswith("sk-ant")
    assert "1234567890abcd" not in by_name["anthropic"].preview
    assert by_name["openai"].configured is False
    assert by_name["openai"].preview == ""
    assert snap.default_provider == "mock"


def test_update_api_keys_writes_env(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    previews = service.update_api_keys(anthropic="sk-anth-1234567890abcd")
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=sk-anth-1234567890abcd" in env_text
    assert previews["anthropic"].startswith("sk-ant")


def test_update_api_keys_delete_flag_removes_line(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=existing\nOTHER=keep\n", encoding="utf-8")
    service = _build_service(tmp_path)
    service.update_api_keys(delete_anthropic=True)
    text = env.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" not in text
    assert "OTHER=keep" in text


def test_update_api_keys_empty_string_skipped(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OTHER=keep\n", encoding="utf-8")
    service = _build_service(tmp_path)
    service.update_api_keys(anthropic="", openai="   ")
    text = env.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" not in text
    assert "OPENAI_API_KEY" not in text
    assert "OTHER=keep" in text


def test_update_models_merges_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "risk:\n  max_loss_per_trade_pct: 0.5\n"
        "analyst:\n  default_provider: mock\n  model_anthropic: old\n",
        encoding="utf-8",
    )
    service = _build_service(tmp_path)
    service.update_models(model_anthropic="claude-3-5-sonnet-latest", default_provider="anthropic")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["risk"]["max_loss_per_trade_pct"] == 0.5
    assert data["analyst"]["model_anthropic"] == "claude-3-5-sonnet-latest"
    assert data["analyst"]["default_provider"] == "anthropic"


def test_update_models_no_updates_skips_write(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    service = _build_service(tmp_path)
    service.update_models()
    assert not cfg.exists()


def test_update_models_rejects_bad_default(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    with pytest.raises(ValueError):
        service.update_models(default_provider="bogus")  # type: ignore[arg-type]


async def test_test_connection_with_stub_provider(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    result = await service.test_connection("anthropic")
    assert result.ok is True
    assert result.response_preview == "PONG"
    assert result.latency_ms >= 0


async def test_test_connection_handles_missing_key(tmp_path: Path) -> None:
    settings = _settings_with_keys()  # no keys
    env = tmp_path / ".env"
    cfg = tmp_path / "config.yaml"

    def factory(name: ProviderName, _config: AnalystProviderConfig) -> object:
        msg = f"{name.upper()}_API_KEY not configured"
        raise ValueError(msg)

    service = LLMSettingsService(
        settings=settings,
        env_path=env,
        config_path=cfg,
        provider_factory=factory,  # type: ignore[arg-type]
        reload_settings=lambda: settings,
    )
    result = await service.test_connection("openai")
    assert result.ok is False
    assert result.error is not None
    assert "OPENAI" in result.error


# ---------------------------------------------------------------------------
# HTTP-level smoke tests for the routes
# ---------------------------------------------------------------------------


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("analyst:\n  default_provider: mock\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n", encoding="utf-8")
    settings = AppSettings.model_validate({})
    settings = settings.model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "LLM_TEST_KILL",
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


def test_llm_page_renders(client: TestClient) -> None:
    response = client.get("/llm")
    assert response.status_code == 200
    body = response.text
    assert "LLM providers" in body
    # Provider names render in lowercase (CSS "capitalize" handles display).
    assert 'data-provider="anthropic"' in body
    assert 'data-provider="openai"' in body
    assert 'data-provider="google"' in body
    assert "ANTHROPIC_API_KEY" in body
    assert "default provider" in body


def test_llm_keys_writes_env(client: TestClient, app_state: AppState) -> None:
    response = client.post(
        "/api/llm/keys",
        json={"openai": "sk-openai-secret-very-long"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    text = app_state.env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-openai-secret-very-long" in text
    # The route reloads via AppSettings.from_yaml, which does NOT read the tmp
    # .env file (pydantic-settings binds to the project .env), so the preview
    # may not reflect the new key here. The file-content assertion above is the
    # authoritative check; this just asserts the response shape.
    assert isinstance(payload["previews"], dict)


def test_llm_keys_delete(client: TestClient, app_state: AppState) -> None:
    app_state.env_path.write_text("ANTHROPIC_API_KEY=oldkey\n", encoding="utf-8")
    response = client.post("/api/llm/keys", json={"delete_anthropic": True})
    assert response.status_code == 200
    assert "ANTHROPIC_API_KEY" not in app_state.env_path.read_text(encoding="utf-8")


def test_llm_models_writes_config(client: TestClient, app_state: AppState) -> None:
    response = client.post(
        "/api/llm/models",
        json={"model_openai": "gpt-4o", "default_provider": "openai"},
    )
    assert response.status_code == 200
    data = yaml.safe_load(app_state.config_path.read_text(encoding="utf-8"))
    assert data["analyst"]["model_openai"] == "gpt-4o"
    assert data["analyst"]["default_provider"] == "openai"


def test_llm_test_endpoint_with_stub(client: TestClient) -> None:
    with patch(
        "dashboard.services.llm_settings._default_provider_factory",
        lambda name, _cfg: MockLLMProvider("PONG", name=name),
    ):
        response = client.post("/api/llm/test", json={"provider": "anthropic"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["response_preview"] == "PONG"
