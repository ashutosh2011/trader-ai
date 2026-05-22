"""Backtests page — list past runs + detail page with equity curve."""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import base_context, get_backtest_runner, get_templates
from dashboard.state import AppState

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/backtests", response_class=HTMLResponse)
async def backtests(request: Request) -> Response:
    """List past backtest runs + render the 'run new' form."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    runner = get_backtest_runner(state)
    runs = await asyncio.to_thread(runner.list_runs, 50)
    strategies = await asyncio.to_thread(runner.list_strategies)
    ctx = base_context(request, active_nav="backtests")
    ctx.update(
        {
            "runs": runs,
            "strategies": strategies,
            "kite_configured": state.settings.kite_configured(),
        }
    )
    return templates.TemplateResponse(request, "backtests.html", ctx)


@router.get("/backtests/{run_id}", response_class=HTMLResponse)
async def backtest_detail(request: Request, run_id: str) -> Response:
    """Render the equity curve, metrics, and trades for one run."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    runner = get_backtest_runner(state)
    detail = await asyncio.to_thread(runner.get_run, run_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"backtest run not found: {run_id}",
        )
    ctx = base_context(request, active_nav="backtests")
    ctx.update(
        {
            "detail": detail,
            "equity_json": json.dumps(detail.equity_curve),
        }
    )
    return templates.TemplateResponse(request, "backtest_detail.html", ctx)
