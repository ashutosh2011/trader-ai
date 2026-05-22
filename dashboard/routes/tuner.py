"""Strategy tuner pages and API (post-trade LLM recommendations)."""

from __future__ import annotations

import asyncio
import json
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from dashboard.routes._common import base_context, get_templates
from dashboard.services.tuner_service import (
    PROVIDER_OPTIONS,
    TunerService,
    action_label,
)
from dashboard.state import AppState

logger = structlog.get_logger(__name__)
router = APIRouter()

STATUS_BADGE: dict[str, str] = {
    "ok": "bg-green-900/40 text-green-300 border-green-700/60",
    "fallback_transport": "bg-yellow-900/40 text-yellow-200 border-yellow-700/60",
    "fallback_parse_error": "bg-red-900/40 text-red-200 border-red-700/60",
    "fallback_unexpected": "bg-red-900/40 text-red-200 border-red-700/60",
}

REC_STATUS_BADGE: dict[str, str] = {
    "pending": "bg-blue-900/40 text-blue-200 border-blue-700/60",
    "applied": "bg-green-900/40 text-green-300 border-green-700/60",
    "rejected": "bg-gray-800 text-gray-400 border-gray-600",
}


class TunerRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "anthropic", "google", "stub"]
    lookback_days: int = Field(default=30, ge=1, le=365)
    notes: str = ""


def _service(state: AppState) -> TunerService:
    return TunerService(state)


@router.get("/tuner", response_class=HTMLResponse)
async def tuner_page(request: Request) -> Response:
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    store = _service(state).tuning_store()
    active = _service(state).active_store()
    runs = await asyncio.to_thread(store.list_runs, 20)
    configs = await asyncio.to_thread(active.list_all)
    ctx = base_context(request, active_nav="tuner")
    ctx.update(
        {
            "runs": runs,
            "active_configs": configs,
            "providers": list(PROVIDER_OPTIONS),
            "status_badge": STATUS_BADGE,
        }
    )
    return templates.TemplateResponse(request, "tuner.html", ctx)


@router.get("/tuner/{run_id}", response_class=HTMLResponse)
async def tuner_detail(request: Request, run_id: str) -> Response:
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    detail = await asyncio.to_thread(_service(state).tuning_store().get_run, run_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    ctx = base_context(request, active_nav="tuner")
    ctx.update(
        {
            "detail": detail,
            "plan_json": json.dumps(
                detail.plan.model_dump(mode="json"),
                indent=2,
            ),
            "status_badge": STATUS_BADGE,
            "rec_status_badge": REC_STATUS_BADGE,
            "action_label": action_label,
        }
    )
    return templates.TemplateResponse(request, "tuner_detail.html", ctx)


@router.post("/api/tuner/run")
async def tuner_run(request: Request, body: TunerRunRequest) -> dict[str, str]:
    state: AppState = request.app.state.dashboard
    async with state.write_lock:
        try:
            record = await _service(state).run(
                provider_name=body.provider,
                notes=body.notes,
                lookback_days=body.lookback_days,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    return {"run_id": record.run_id}


@router.post("/api/tuner/recommendations/{rec_id}/apply")
async def tuner_apply(request: Request, rec_id: str) -> dict[str, bool]:
    state: AppState = request.app.state.dashboard
    async with state.write_lock:
        try:
            await asyncio.to_thread(_service(state).apply_recommendation, rec_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    return {"ok": True}


@router.post("/api/tuner/recommendations/{rec_id}/reject")
async def tuner_reject(request: Request, rec_id: str) -> dict[str, bool]:
    state: AppState = request.app.state.dashboard
    async with state.write_lock:
        try:
            await asyncio.to_thread(_service(state).reject_recommendation, rec_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    return {"ok": True}
