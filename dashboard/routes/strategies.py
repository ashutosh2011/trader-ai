"""Strategies page — list registered strategies + enable/disable toggle."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response

from dashboard.routes._common import base_context, get_strategy_state, get_templates
from dashboard.state import AppState
from strategies.registry import get_strategy, list_strategies

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/strategies", response_class=HTMLResponse)
async def strategies(request: Request) -> Response:
    """Render the strategies table."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    svc = get_strategy_state(state)
    persisted = await asyncio.to_thread(svc.list_all)
    items: list[dict[str, Any]] = []
    for strategy_id in list_strategies():
        strategy_cls = get_strategy(strategy_id)
        enabled = persisted[strategy_id].enabled if strategy_id in persisted else True
        required = [
            indicator.param_key()
            for indicator in getattr(strategy_cls, "required_indicators", []) or []
            if hasattr(indicator, "param_key")
        ]
        items.append(
            {
                "id": strategy_id,
                "timeframe": getattr(strategy_cls, "timeframe", "?"),
                "enabled": enabled,
                "required_indicators": required,
                "doc": (strategy_cls.__doc__ or "").strip().splitlines()[0]
                if strategy_cls.__doc__
                else "",
            }
        )
    ctx = base_context(request, active_nav="strategies")
    ctx.update({"items": items})
    return templates.TemplateResponse(request, "strategies.html", ctx)
