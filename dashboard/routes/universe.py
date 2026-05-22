"""Universe management page + JSON API for symbol CRUD."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import Response

from dashboard.routes._common import base_context, get_templates
from dashboard.services.universe_io import UniverseIO, UniverseIOError
from dashboard.state import AppState

logger = structlog.get_logger(__name__)
router = APIRouter()


def _service() -> UniverseIO:
    return UniverseIO()


class UniverseAddRequest(BaseModel):
    """Body for ``POST /api/universe/add``."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=32)
    exchange: str = Field(default="NSE", min_length=1, max_length=16)
    instrument_token: int | None = Field(default=None, ge=1)

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, value: str) -> str:
        text = value.strip().upper()
        if not text:
            msg = "symbol must be non-empty"
            raise ValueError(msg)
        return text

    @field_validator("exchange")
    @classmethod
    def upper_exchange(cls, value: str) -> str:
        return value.strip().upper()


class UniverseUpdateRequest(BaseModel):
    """Body for ``POST /api/universe/update/{symbol}``."""

    model_config = ConfigDict(extra="forbid")

    exchange: str | None = Field(default=None, min_length=1, max_length=16)
    instrument_token: int | None = Field(default=None, ge=1)

    @field_validator("exchange")
    @classmethod
    def upper_exchange(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().upper()


class UniverseSeedRequest(BaseModel):
    """Body for ``POST /api/universe/seed``."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=15, ge=1, le=100)


@router.get("/universe", response_class=HTMLResponse)
async def universe_page(request: Request) -> Response:
    """Render the universe editor."""
    state: AppState = request.app.state.dashboard
    templates = get_templates(request)
    service = _service()
    try:
        universe = await asyncio.to_thread(service.load_universe_editable)
        symbols = list(universe.symbols)
        load_error: str | None = None
    except (FileNotFoundError, UniverseIOError) as exc:
        symbols = []
        load_error = str(exc)
    ctx = base_context(request, active_nav="universe")
    ctx.update(
        {
            "symbols": symbols,
            "load_error": load_error,
            "universe_path": str(service.path),
            "kite_configured": state.settings.kite_configured(),
        }
    )
    return templates.TemplateResponse(request, "universe.html", ctx)


@router.post("/api/universe/add")
async def universe_add(request: Request, body: UniverseAddRequest) -> dict[str, Any]:
    """Append a new symbol entry and persist."""
    state: AppState = request.app.state.dashboard
    service = _service()
    async with state.write_lock:
        try:
            universe = await asyncio.to_thread(
                service.add_symbol,
                symbol=body.symbol,
                exchange=body.exchange,
                instrument_token=body.instrument_token,
            )
        except UniverseIOError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    return {"ok": True, "size": len(universe.symbols)}


@router.post("/api/universe/update/{symbol}")
async def universe_update(
    request: Request,
    symbol: str,
    body: UniverseUpdateRequest,
) -> dict[str, Any]:
    """Update exchange or instrument_token for ``symbol``."""
    state: AppState = request.app.state.dashboard
    service = _service()
    async with state.write_lock:
        try:
            universe = await asyncio.to_thread(
                service.update_symbol,
                symbol=symbol,
                exchange=body.exchange,
                instrument_token=body.instrument_token,
            )
        except UniverseIOError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    return {"ok": True, "size": len(universe.symbols)}


@router.post("/api/universe/delete/{symbol}")
async def universe_delete(request: Request, symbol: str) -> dict[str, Any]:
    """Remove ``symbol`` from the universe."""
    state: AppState = request.app.state.dashboard
    service = _service()
    async with state.write_lock:
        try:
            universe = await asyncio.to_thread(service.delete_symbol, symbol)
        except UniverseIOError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    return {"ok": True, "size": len(universe.symbols)}


@router.post("/api/universe/seed")
async def universe_seed(request: Request, body: UniverseSeedRequest) -> dict[str, Any]:
    """Append unseen popular NSE symbols up to ``limit``."""
    state: AppState = request.app.state.dashboard
    service = _service()
    async with state.write_lock:
        try:
            added = await asyncio.to_thread(service.seed_popular, limit=body.limit)
        except UniverseIOError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    return {"ok": True, "added": added}
