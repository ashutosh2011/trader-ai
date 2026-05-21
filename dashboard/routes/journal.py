"""Journal page — live tail of the JSONL trading journal."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import base_context, get_journal_reader, get_templates
from dashboard.state import AppState

logger = structlog.get_logger(__name__)
router = APIRouter()

_KNOWN_EVENT_TYPES = ["signal", "verdict", "risk_decision", "order"]


@router.get("/journal", response_class=HTMLResponse)
async def journal(
    request: Request,
    event: str | None = None,
    symbol: str | None = None,
    limit: int = 200,
) -> Response:
    """Render the journal page — paused by default, polled every 3s."""
    app_state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    reader = get_journal_reader(app_state)
    result = await asyncio.to_thread(
        reader.tail,
        limit=limit,
        event_types=[event] if event else None,
        symbol=symbol,
    )
    ctx = base_context(request, active_nav="journal")
    ctx.update(
        {
            "entries": result.entries,
            "last_ts": result.last_ts,
            "event_types": _KNOWN_EVENT_TYPES,
            "filter_event": event or "",
            "filter_symbol": symbol or "",
            "limit": limit,
            "exists": reader.exists(),
            "path": str(reader.path) if reader.path else None,
        }
    )
    return templates.TemplateResponse(request, "journal.html", ctx)


@router.get("/_partials/journal/tail", response_class=HTMLResponse)
async def journal_tail_partial(
    request: Request,
    since_ts: str | None = None,
    event: str | None = None,
    symbol: str | None = None,
    limit: int = 200,
) -> Response:
    """HTMX partial: append new entries since ``since_ts``."""
    app_state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    reader = get_journal_reader(app_state)
    result = await asyncio.to_thread(
        reader.tail,
        limit=limit,
        since_ts=since_ts,
        event_types=[event] if event else None,
        symbol=symbol,
    )
    return templates.TemplateResponse(
        request,
        "partials/journal_rows.html",
        {
            "request": request,
            "entries": result.entries,
            "last_ts": result.last_ts,
        },
    )
