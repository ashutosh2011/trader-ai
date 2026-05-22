"""Screener page + JSON API for triggering and viewing runs.

Mirrors the backtest module structure: a list page with a "Run new"
form, a JSON ``/api/screener/run`` endpoint that returns a run id, and
a detail page that pretty-prints the formula plus picks. The screener
is intentionally read-only — nothing here wires into the trading loop.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from kiteconnect.exceptions import KiteException
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from dashboard.routes._common import base_context, get_templates
from dashboard.services.screener_service import (
    PROVIDER_OPTIONS,
    ScreenerProviderName,
    ScreenerService,
    filter_to_sentence,
)
from dashboard.state import AppState
from screener.prompt import MarketContext

logger = structlog.get_logger(__name__)
router = APIRouter()

IST = ZoneInfo("Asia/Kolkata")

STATUS_BADGE: dict[str, str] = {
    "ok": "bg-green-900/40 text-green-300 border-green-700/60",
    "fallback_transport": "bg-yellow-900/40 text-yellow-200 border-yellow-700/60",
    "fallback_parse_error": "bg-red-900/40 text-red-200 border-red-700/60",
    "fallback_unexpected": "bg-red-900/40 text-red-200 border-red-700/60",
}


class ScreenerRunRequest(BaseModel):
    """Body for ``POST /api/screener/run``."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "anthropic", "google", "stub"]
    fetch_missing: bool = False
    bars_back: int = Field(default=200, ge=10, le=2000)
    market_context_notes: str = ""
    recent_index_summary: str = "No external index summary provided."


def _service(state: AppState) -> ScreenerService:
    return ScreenerService(state.screener_store(), state.settings)


@router.get("/screener", response_class=HTMLResponse)
async def screener_page(request: Request) -> Response:
    """List recent runs + render the run-new form."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    store = state.screener_store()
    runs = await asyncio.to_thread(store.list_runs, 20)
    badge_map = STATUS_BADGE
    ctx = base_context(request, active_nav="screener")
    ctx.update(
        {
            "runs": runs,
            "providers": list(PROVIDER_OPTIONS),
            "kite_configured": state.settings.kite_configured(),
            "status_badge": badge_map,
        }
    )
    return templates.TemplateResponse(request, "screener.html", ctx)


@router.get("/screener/{run_id}", response_class=HTMLResponse)
async def screener_detail(request: Request, run_id: str) -> Response:
    """Render the picks and full formula for one run."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    store = state.screener_store()
    detail = await asyncio.to_thread(store.get_run, run_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"screener run not found: {run_id}",
        )
    formula_payload = detail.formula.model_dump()
    filter_payloads = list(formula_payload.get("filters", []))
    sentences = [filter_to_sentence(filter_payload) for filter_payload in filter_payloads]
    ctx = base_context(request, active_nav="screener")
    ctx.update(
        {
            "detail": detail,
            "formula_json_pretty": json.dumps(formula_payload, indent=2, default=str),
            "filter_sentences": sentences,
            "filter_payloads": filter_payloads,
            "status_badge": STATUS_BADGE,
        }
    )
    return templates.TemplateResponse(request, "screener_detail.html", ctx)


@router.post("/api/screener/run")
async def screener_run(request: Request, body: ScreenerRunRequest) -> dict[str, Any]:
    """Run a screener pass and return ``{run_id}`` once persisted."""
    app_state: AppState = request.app.state.dashboard
    service = _service(app_state)
    provider_name: ScreenerProviderName = body.provider
    market_context = MarketContext(
        as_of=datetime.now(tz=IST),
        recent_index_summary=body.recent_index_summary,
        notes=body.market_context_notes,
    )
    async with app_state.write_lock:
        try:
            record = await service.run(
                provider_name=provider_name,
                market_context=market_context,
                fetch_missing=body.fetch_missing,
                bars_back=body.bars_back,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except KiteException as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Kite request failed: {exc}",
            ) from exc
    return {
        "run_id": record.id,
        "status": record.meta.status,
        "passed_count": len(record.results),
        "eligible_count": record.eligible_count,
        "universe_size": record.universe_size,
    }
