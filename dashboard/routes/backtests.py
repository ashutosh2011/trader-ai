"""Backtests page — list past runs + detail page with equity curve."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import (
    base_context,
    get_backtest_runner,
    get_sweep_runner,
    get_templates,
)
from dashboard.services.strategy_schemas import all_schemas, to_json_dict
from dashboard.services.sweep_runner import SweepStatus
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
    instruments_service = state.instruments()
    instruments_status = await asyncio.to_thread(instruments_service.status)
    lookup = SymbolLookupService(instruments_service)
    # Embed the full NSE universe so the picker lists every stock, not just
    # the first alphabetical page. The JS prefers embedded options over a
    # fetch, so a small cap here silently truncated the dropdown.
    symbols = await asyncio.to_thread(lookup.list_symbols, limit=100000)
    sweeps = await asyncio.to_thread(_list_recent_sweeps, state)
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
            "instruments_status": instruments_status,
            "instruments_status_json": json.dumps(instruments_status),
            "sweeps": sweeps,
        }
    )
    return templates.TemplateResponse(request, "backtests.html", ctx)


@router.get("/backtests/sweep/new", response_class=HTMLResponse)
async def backtests_sweep_new(request: Request) -> Response:
    """Render the multi-symbol × strategy-grid sweep form."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    schemas = all_schemas()
    instruments_status = await asyncio.to_thread(state.instruments().status)
    ctx = base_context(request, active_nav="backtests")
    ctx.update(
        {
            "schemas": schemas,
            "schema_json": json.dumps(to_json_dict()),
            "instruments_status": instruments_status,
            "instruments_status_json": json.dumps(instruments_status),
            "kite_configured": state.settings.kite_configured(),
        }
    )
    return templates.TemplateResponse(
        request, "backtest_sweep_new.html", ctx
    )


@router.get("/backtests/sweep/{sweep_id}", response_class=HTMLResponse)
async def backtests_sweep_detail(request: Request, sweep_id: str) -> Response:
    """Render the sweep progress page + leaderboard / heatmap when done."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    sweep_runner = get_sweep_runner(state)
    snapshot = await asyncio.to_thread(sweep_runner.status, sweep_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"sweep not found: {sweep_id}",
        )
    config = await asyncio.to_thread(sweep_runner.get_config, sweep_id)
    leaderboard = []
    heatmap: dict[str, Any] = {"symbols": [], "strategies": [], "cells": []}
    if snapshot.status == "done":
        leaderboard = await asyncio.to_thread(sweep_runner.leaderboard, sweep_id)
        heatmap = await asyncio.to_thread(sweep_runner.heatmap, sweep_id)
    ctx = base_context(request, active_nav="backtests")
    ctx.update(
        {
            "snapshot": snapshot,
            "config": config,
            "leaderboard": leaderboard,
            "heatmap": heatmap,
            "heatmap_json": json.dumps(heatmap),
            "snapshot_json": json.dumps(_snapshot_payload(snapshot)),
        }
    )
    return templates.TemplateResponse(
        request, "backtest_sweep_detail.html", ctx
    )


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


def _list_recent_sweeps(state: AppState) -> list[dict[str, Any]]:
    rows = state.dashboard_conn().execute(
        "SELECT id, label, created_at, status, total, completed, failed "
        "FROM backtest_sweeps ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": str(row[0]),
                "label": str(row[1]),
                "created_at": row[2],
                "status": str(row[3]),
                "total": int(row[4]),
                "completed": int(row[5]),
                "failed": int(row[6]),
            }
        )
    return out


def _snapshot_payload(snapshot: SweepStatus) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "label": snapshot.label,
        "status": snapshot.status,
        "total": snapshot.total,
        "completed": snapshot.completed,
        "failed": snapshot.failed,
        "error": snapshot.error,
        "elapsed_ms": snapshot.elapsed_ms,
        "timeframe": snapshot.timeframe,
        "from_date": snapshot.from_date,
        "to_date": snapshot.to_date,
        "qty": snapshot.qty,
    }
