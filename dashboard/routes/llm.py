"""LLM settings page + JSON API for keys, models, and connection tests."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from dashboard.routes._common import base_context, get_templates
from dashboard.services.llm_settings import (
    MODEL_OPTIONS,
    PROVIDER_NAMES,
    DefaultProviderName,
    LLMSettingsService,
    ProviderName,
)
from dashboard.state import AppState

logger = structlog.get_logger(__name__)
router = APIRouter()


def _service(state: AppState) -> LLMSettingsService:
    """Build a fresh :class:`LLMSettingsService` against the given state."""
    return LLMSettingsService(
        settings=state.settings,
        env_path=state.env_path,
        config_path=state.config_path,
        reload_settings=state.reload_settings,
    )


class LLMKeysRequest(BaseModel):
    """Body for ``POST /api/llm/keys``."""

    model_config = ConfigDict(extra="forbid")

    anthropic: str | None = None
    openai: str | None = None
    google: str | None = None
    delete_anthropic: bool = False
    delete_openai: bool = False
    delete_google: bool = False


class LLMModelsRequest(BaseModel):
    """Body for ``POST /api/llm/models``."""

    model_config = ConfigDict(extra="forbid")

    model_anthropic: str | None = Field(default=None)
    model_openai: str | None = Field(default=None)
    model_google: str | None = Field(default=None)
    default_provider: Literal["anthropic", "openai", "google", "mock"] | None = None


class LLMTestRequest(BaseModel):
    """Body for ``POST /api/llm/test``."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "openai", "google"]


@router.get("/llm", response_class=HTMLResponse)
async def llm_page(request: Request) -> Response:
    """Render the LLM provider settings page."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    service = _service(state)
    snapshot = await asyncio.to_thread(service.read_status)
    ctx = base_context(request, active_nav="llm")
    ctx.update(
        {
            "snapshot": snapshot,
            "provider_names": list(PROVIDER_NAMES),
            "model_options": {name: list(MODEL_OPTIONS[name]) for name in PROVIDER_NAMES},
        }
    )
    return templates.TemplateResponse(request, "llm.html", ctx)


@router.post("/api/llm/keys")
async def llm_keys(request: Request, body: LLMKeysRequest) -> dict[str, Any]:
    """Write provider keys to ``.env`` (or delete them) and return previews."""
    state: AppState = request.app.state.dashboard
    service = _service(state)
    async with state.write_lock:
        try:
            previews = await asyncio.to_thread(
                service.update_api_keys,
                anthropic=body.anthropic,
                openai=body.openai,
                google=body.google,
                delete_anthropic=body.delete_anthropic,
                delete_openai=body.delete_openai,
                delete_google=body.delete_google,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    return {"ok": True, "previews": previews}


@router.post("/api/llm/models")
async def llm_models(request: Request, body: LLMModelsRequest) -> dict[str, Any]:
    """Persist analyst-block model / default-provider preferences."""
    state: AppState = request.app.state.dashboard
    service = _service(state)
    async with state.write_lock:
        try:
            await asyncio.to_thread(
                service.update_models,
                model_anthropic=body.model_anthropic,
                model_openai=body.model_openai,
                model_google=body.model_google,
                default_provider=_cast_default_provider(body.default_provider),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    return {"ok": True}


@router.post("/api/llm/test")
async def llm_test(request: Request, body: LLMTestRequest) -> dict[str, Any]:
    """Run a tiny ping against the selected provider."""
    state: AppState = request.app.state.dashboard
    service = _service(state)
    provider_name: ProviderName = body.provider
    result = await service.test_connection(provider_name)
    return {
        "ok": result.ok,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "response_preview": result.response_preview,
    }


def _cast_default_provider(value: str | None) -> DefaultProviderName | None:
    if value is None:
        return None
    if value not in {"anthropic", "openai", "google", "mock"}:
        msg = f"unknown default_provider: {value}"
        raise ValueError(msg)
    return value  # type: ignore[return-value]
