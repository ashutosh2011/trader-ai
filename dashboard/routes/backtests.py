"""Backtests page — list past runs + detail page with equity curve."""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import base_context, get_backtest_runner, get_templates
from dashboard.services.strategy_schemas import all_schemas, to_json_dict
from dashboard.services.symbol_lookup import SymbolLookupService
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
    schemas = all_schemas()
    lookup = SymbolLookupService()
    symbols = await asyncio.to_thread(lookup.list_symbols)
    ctx = base_context(request, active_nav="backtests")
    ctx.update(
        {
            "runs": runs,
            "strategies": strategies,
            "schemas": schemas,
            "schema_json": json.dumps(to_json_dict()),
            "symbol_options": [s.to_json() for s in symbols],
            "symbol_options_json": json.dumps([s.to_json() for s in symbols]),
            "kite_configured": state.settings.kite_configured(),
        }
    )
    return templates.TemplateResponse(request, "backtests.html", ctx)


@router.get("/backtests/compare/{group_id}", response_class=HTMLResponse)
async def backtest_compare(request: Request, group_id: str) -> Response:
    """Side-by-side metric table + combined equity chart for one group."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    runner = get_backtest_runner(state)
    group = await asyncio.to_thread(runner.get_group, group_id)
    if group is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"backtest group not found: {group_id}",
        )
    # Best performer ranked by realised P&L; ties resolved by Sharpe so
    # two strategies with identical P&L don't flip-flop between renders.
    members_sorted = sorted(
        group.members,
        key=lambda m: (m.summary.total_pnl, m.summary.sharpe),
        reverse=True,
    )
    best = members_sorted[0] if members_sorted else None
    chart_payload = [
        {
            "label": member.summary.strategy,
            "equity": member.equity_curve,
        }
        for member in group.members
    ]
    ctx = base_context(request, active_nav="backtests")
    ctx.update(
        {
            "group": group,
            "members_sorted": members_sorted,
            "best": best,
            "chart_json": json.dumps(chart_payload),
        }
    )
    return templates.TemplateResponse(request, "backtest_compare.html", ctx)


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
