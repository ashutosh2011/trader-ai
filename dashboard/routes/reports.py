"""Reports page — strategy / symbol P&L roll-ups and run highlights."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import base_context, get_templates
from dashboard.services.reports import ReportsService
from dashboard.state import AppState

logger = structlog.get_logger(__name__)
router = APIRouter()


@dataclass(frozen=True)
class _ReportsContext:
    """Aggregate context handed to the template."""

    by_strategy: list[Any]
    by_symbol: list[Any]
    winners: list[Any]
    losers: list[Any]


@router.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request) -> Response:
    """Render the reports dashboard."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    service = ReportsService(state.dashboard_conn())
    payload = await asyncio.to_thread(_collect, service)
    ctx = base_context(request, active_nav="reports")
    ctx.update(
        {
            "by_strategy": payload.by_strategy,
            "by_symbol": payload.by_symbol,
            "winners": payload.winners,
            "losers": payload.losers,
        }
    )
    return templates.TemplateResponse(request, "reports.html", ctx)


def _collect(service: ReportsService) -> _ReportsContext:
    return _ReportsContext(
        by_strategy=service.by_strategy(),
        by_symbol=service.by_symbol(),
        winners=service.top_winners(),
        losers=service.top_losers(),
    )
