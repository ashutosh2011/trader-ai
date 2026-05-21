"""Config page — YAML editor with validate / save."""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import base_context, get_templates
from dashboard.services.config_io import read_config_text
from dashboard.state import AppState

logger = structlog.get_logger(__name__)
router = APIRouter()

_EXAMPLE_PATH = Path("config/config.example.yaml")


@router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request) -> Response:
    """Render the config editor pre-filled with the current YAML."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    text = await asyncio.to_thread(read_config_text, state.config_path, _EXAMPLE_PATH)
    ctx = base_context(request, active_nav="config")
    ctx.update(
        {
            "config_text": text,
            "config_path": str(state.config_path),
            "env_path": str(state.env_path),
        }
    )
    return templates.TemplateResponse(request, "config.html", ctx)
