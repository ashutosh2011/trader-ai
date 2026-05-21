"""Orders page — filtered + paginated table + admin marks."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import base_context, get_orders_service, get_templates
from dashboard.state import AppState
from execution.order_state import OrderState

logger = structlog.get_logger(__name__)
router = APIRouter()

_ALL_STATES = ["ALL", *[state.value for state in OrderState]]


@router.get("/orders", response_class=HTMLResponse)
async def orders(
    request: Request,
    state: str = "ALL",
    symbol: str | None = None,
    page: int = 1,
    per_page: int = 50,
) -> Response:
    """Render the orders table with filters + pagination."""
    app_state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    service = get_orders_service(app_state)
    result = await asyncio.to_thread(
        service.page,
        state=state,
        symbol=symbol,
        page=page,
        per_page=per_page,
    )
    ctx = base_context(request, active_nav="orders")
    ctx.update(
        {
            "result": result,
            "filter_state": state,
            "filter_symbol": symbol or "",
            "all_states": _ALL_STATES,
            "page": page,
            "per_page": per_page,
        }
    )
    return templates.TemplateResponse(request, "orders.html", ctx)
