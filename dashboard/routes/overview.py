"""Overview page — kill switch, today's PnL, quick actions."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import (
    base_context,
    get_journal_reader,
    get_kill_service,
    get_orders_service,
    get_templates,
)
from dashboard.state import AppState

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def overview(request: Request) -> Response:
    """Render the overview page with one HTMX-polled partial."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    snapshot = await asyncio.to_thread(_snapshot, state)
    ctx = base_context(request, active_nav="overview")
    ctx.update({"snapshot": snapshot})
    return templates.TemplateResponse(request, "overview.html", ctx)


@router.get("/_partials/overview/state", response_class=HTMLResponse)
async def overview_state_partial(request: Request) -> Response:
    """HTMX partial: re-render the live counters every 5s."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    snapshot = await asyncio.to_thread(_snapshot, state)
    ctx = base_context(request, active_nav="overview")
    ctx.update({"snapshot": snapshot})
    return templates.TemplateResponse(request, "partials/overview_state.html", ctx)


def _snapshot(state: AppState) -> dict[str, Any]:
    """Build the dict consumed by ``overview.html`` and its partial."""
    kill = get_kill_service(state)
    journal = get_journal_reader(state)
    orders = get_orders_service(state)
    open_records = orders.list_open()
    counts = journal.event_counts_today()
    last_events = journal.tail(limit=5)
    return {
        "kill_active": kill.is_active(),
        "kill_file": str(state.settings.kill_switch_file),
        "today_realized_pnl": journal.today_realized_pnl(),
        "open_positions": len(open_records),
        "signals_today": counts.get("signal", 0),
        "vetos_today": counts.get("verdict", 0),
        "orders_today": counts.get("order", 0),
        "kite_configured": state.settings.kite_configured(),
        "last_events": [
            {"ts": e.ts, "event": e.event, "payload": e.payload}
            for e in last_events.entries
        ],
    }
