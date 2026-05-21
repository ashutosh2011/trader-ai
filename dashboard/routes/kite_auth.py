"""Kite login page — build login URL + exchange request token."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import base_context, get_kite_service, get_templates
from dashboard.state import AppState

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/kite", response_class=HTMLResponse)
async def kite_page(request: Request) -> Response:
    """Render the Kite login + token status panel."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    service = get_kite_service(state)
    configured = service.is_configured()
    login_url: str | None = None
    if configured:
        try:
            login_url = await asyncio.to_thread(service.login_url)
        except ValueError as exc:
            logger.warning("dashboard_kite_login_url_failed", error=str(exc))
            login_url = None
    status = await asyncio.to_thread(service.token_status)
    ctx = base_context(request, active_nav="kite")
    ctx.update(
        {
            "configured": configured,
            "login_url": login_url,
            "status": status,
            "env_path": str(state.env_path),
            "api_key_present": bool(state.settings.kite.api_key),
            "api_secret_present": bool(state.settings.kite.api_secret),
        }
    )
    return templates.TemplateResponse(request, "kite.html", ctx)
