"""Live page — open positions, recent orders, intraday PnL chart."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import (
    base_context,
    get_journal_reader,
    get_orders_service,
    get_templates,
)
from dashboard.state import AppState
from execution.order_state import OrderState

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/live", response_class=HTMLResponse)
async def live(request: Request) -> Response:
    """Render the live page."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    snapshot = await asyncio.to_thread(_snapshot, state)
    ctx = base_context(request, active_nav="live")
    ctx.update(snapshot)
    return templates.TemplateResponse(request, "live.html", ctx)


@router.get("/_partials/live/state", response_class=HTMLResponse)
async def live_partial(request: Request) -> Response:
    """Partial that re-renders the position + orders + PnL chart."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    snapshot = await asyncio.to_thread(_snapshot, state)
    ctx = base_context(request, active_nav="live")
    ctx.update(snapshot)
    return templates.TemplateResponse(request, "partials/live_state.html", ctx)


def _snapshot(state: AppState) -> dict[str, Any]:
    orders = get_orders_service(state)
    open_records = orders.list_open()
    all_records = orders.list_all()
    recent_records = sorted(all_records, key=lambda r: r.created_at, reverse=True)[:20]
    pnl_series = _intraday_pnl_series(state)
    return {
        "positions": open_records,
        "recent_orders": recent_records,
        "kite_configured": state.settings.kite_configured(),
        "pnl_chart_json": json.dumps(pnl_series),
        "order_states_for_admin": [
            OrderState.PENDING_ENTRY.value,
            OrderState.ENTERED.value,
        ],
    }


def _intraday_pnl_series(state: AppState) -> list[dict[str, Any]]:
    """Compute a per-event running-PnL series from today's journal events.

    The journal stores each ``order`` event with ``pnl`` (paper broker
    close) — we sum cumulatively in ts order. Returns an empty list when
    there is no journal.
    """
    journal = get_journal_reader(state)
    if not journal.exists():
        return []
    today = datetime.now().astimezone().date().isoformat()
    entries = journal.tail(limit=10_000, event_types=["order"])
    series: list[dict[str, Any]] = []
    running = 0.0
    sorted_entries = sorted(entries.entries, key=lambda e: e.ts)
    for entry in sorted_entries:
        if not entry.ts.startswith(today):
            continue
        order = entry.payload.get("order")
        pnl_value = order.get("pnl") if isinstance(order, dict) else entry.payload.get("pnl")
        if not isinstance(pnl_value, int | float):
            continue
        running += float(pnl_value)
        series.append({"ts": entry.ts, "pnl": running})
    return series
