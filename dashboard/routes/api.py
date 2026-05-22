"""JSON API endpoints called by HTMX widgets and by the operator directly.

All endpoints are async. CPU-blocking calls (DuckDB queries, the synchronous
backtest engine) are dispatched onto a worker thread with
:func:`asyncio.to_thread` so the event loop stays responsive when the
single-user UI fires several polling requests at once.

TRADEOFF: We intentionally keep the API thin — most reads are also
available as HTML partials served by the page routes. The JSON endpoints
exist for the dashboard's own widgets and for ad-hoc curl debugging.
There is no auth, see :mod:`dashboard` package docstring.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from kiteconnect.exceptions import KiteException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dashboard.routes._common import (
    base_context,
    format_issues,
    get_backtest_runner,
    get_journal_reader,
    get_kill_service,
    get_kite_service,
    get_orders_service,
    get_strategy_state,
)
from dashboard.services.config_io import save_yaml, validate_yaml
from dashboard.services.journal_reader import JournalReader
from dashboard.services.kill_switch import KillSwitchService
from dashboard.state import AppState
from execution.broker import FlattenIncomplete
from execution.kite import KiteBroker
from execution.order_state import OrderRecord, OrderState

router = APIRouter(prefix="/api", tags=["api"])


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


class KillToggleRequest(BaseModel):
    """Body for ``POST /api/kill/toggle``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool


class FlattenRequest(BaseModel):
    """Body for ``POST /api/flatten`` — requires the literal confirm token."""

    model_config = ConfigDict(extra="forbid")

    confirm: str

    @field_validator("confirm")
    @classmethod
    def must_match_token(cls, value: str) -> str:
        """Reject the request unless the operator typed ``FLATTEN``."""
        if value != "FLATTEN":
            msg = "confirm must equal 'FLATTEN'"
            raise ValueError(msg)
        return value


class BacktestRunRequest(BaseModel):
    """Body for ``POST /api/backtest/run``."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    symbol: str = "SYNTH"
    bars_count: int = Field(default=500, ge=10, le=50_000)
    qty: int = Field(default=1, ge=1)
    seed: int = 42
    data_source: Literal["synthetic", "kite"] = "synthetic"
    instrument_token: int | None = Field(default=None, gt=0)
    timeframe: str | None = None
    from_date: str | None = None
    to_date: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ConfigPayload(BaseModel):
    """Body for ``POST /api/config/{validate,save}``."""

    model_config = ConfigDict(extra="forbid")

    yaml: str


class KiteExchangeRequest(BaseModel):
    """Body for ``POST /api/kite/exchange``."""

    model_config = ConfigDict(extra="forbid")

    request_token: str


class OrderMarkRequest(BaseModel):
    """Body for ``POST /api/orders/{id}/mark``."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["FAILED", "CANCELLED"]
    reason: str = "marked_by_dashboard"


# ---------------------------------------------------------------------------
# read endpoints
# ---------------------------------------------------------------------------


@router.get("/overview/state")
async def overview_state(request: Request) -> dict[str, Any]:
    """Return the overview snapshot used by the HTMX poller."""
    state: AppState = request.app.state.dashboard
    kill = get_kill_service(state)
    journal = get_journal_reader(state)
    base = base_context(request, active_nav="overview")
    snapshot = await asyncio.to_thread(_overview_snapshot, state, kill, journal)
    snapshot.update({"kill_banner": base["kill_banner"], "now": base["now"].isoformat()})
    return snapshot


@router.get("/positions")
async def positions(request: Request) -> dict[str, Any]:
    """Return the broker-reported open positions (paper or kite-derived)."""
    state: AppState = request.app.state.dashboard
    orders = get_orders_service(state)
    open_records = await asyncio.to_thread(orders.list_open)
    return {
        "positions": [_position_dict(record) for record in open_records],
        "count": len(open_records),
    }


@router.get("/orders")
async def orders_page(
    request: Request,
    state: str = "ALL",
    symbol: str | None = None,
    page: int = 1,
    per_page: int = 50,
) -> dict[str, Any]:
    """Filtered + paginated order list (defaults to first 50)."""
    app_state: AppState = request.app.state.dashboard
    orders = get_orders_service(app_state)
    result = await asyncio.to_thread(
        orders.page,
        state=state,
        symbol=symbol,
        page=page,
        per_page=per_page,
    )
    return {
        "rows": [_order_dict(record) for record in result.rows],
        "page": result.page,
        "per_page": result.per_page,
        "total": result.total,
        "total_pages": result.total_pages,
    }


@router.get("/journal/tail")
async def journal_tail(
    request: Request,
    since_ts: str | None = None,
    limit: int = 100,
    event: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Tail the JSONL journal, filtered + bounded."""
    app_state: AppState = request.app.state.dashboard
    reader = get_journal_reader(app_state)
    event_types = [event] if event else None
    result = await asyncio.to_thread(
        reader.tail,
        limit=limit,
        since_ts=since_ts,
        event_types=event_types,
        symbol=symbol,
    )
    return {
        "entries": [
            {"ts": entry.ts, "event": entry.event, "payload": entry.payload}
            for entry in result.entries
        ],
        "last_ts": result.last_ts,
        "exists": reader.exists(),
    }


# ---------------------------------------------------------------------------
# write endpoints
# ---------------------------------------------------------------------------


@router.post("/kill/toggle")
async def kill_toggle(request: Request, body: KillToggleRequest) -> dict[str, Any]:
    """Engage or disarm the kill switch."""
    app_state: AppState = request.app.state.dashboard
    kill = get_kill_service(app_state)
    async with app_state.write_lock:
        active = await asyncio.to_thread(kill.set, body.enabled)
    return {"active": active}


@router.post("/flatten")
async def flatten(request: Request, body: FlattenRequest) -> dict[str, Any]:
    """Square-off every open Kite position at market.

    Returns 400 (Bad Request) when no Kite broker is configured — the
    dashboard refuses to "flatten" the paper broker because the paper
    broker has no real positions outside the live loop's in-memory state.
    """
    del body  # validated for the confirm token
    app_state: AppState = request.app.state.dashboard
    if not app_state.settings.kite_configured():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="no_kite_broker_configured",
        )
    async with app_state.write_lock:
        try:
            await asyncio.to_thread(_flatten_kite, app_state)
        except FlattenIncomplete as exc:
            return {
                "ok": False,
                "open_positions": [p.symbol for p in exc.open_positions],
                "attempts": exc.attempts,
            }
    return {"ok": True}


@router.post("/backtest/run")
async def backtest_run(request: Request, body: BacktestRunRequest) -> dict[str, Any]:
    """Run a backtest synchronously and return the new run id."""
    app_state: AppState = request.app.state.dashboard
    runner = get_backtest_runner(app_state)
    try:
        run_id = await asyncio.to_thread(
            runner.run,
            strategy_id=body.strategy,
            symbol=body.symbol,
            bars_count=body.bars_count,
            params=body.params,
            qty=body.qty,
            seed=body.seed,
            data_source=body.data_source,
            instrument_token=body.instrument_token,
            timeframe=body.timeframe,
            from_date=body.from_date,
            to_date=body.to_date,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"strategy_not_registered: {exc}",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except KiteException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_kite_backtest_error_message(exc),
        ) from exc
    return {"id": run_id}


def _kite_backtest_error_message(exc: KiteException) -> str:
    """Return operator-friendly guidance for Kite historical-data failures."""
    text = str(exc)
    if "api_key" in text or "access_token" in text or "Token" in type(exc).__name__:
        return (
            f"Kite rejected the historical-data request: {text}. "
            "Refresh today's access token from the Kite page, verify the API key "
            "matches the app that generated the token, then retry."
        )
    return f"Kite historical-data request failed: {text}"


@router.post("/config/validate")
async def config_validate(body: ConfigPayload) -> dict[str, Any]:
    """Return a validation result without writing anything."""
    result = await asyncio.to_thread(validate_yaml, body.yaml)
    return {"ok": result.ok, "issues": format_issues(result.issues)}


@router.post("/config/save")
async def config_save(request: Request, body: ConfigPayload) -> dict[str, Any]:
    """Validate then write the YAML config, backing the prior file up."""
    app_state: AppState = request.app.state.dashboard
    async with app_state.write_lock:
        result = await asyncio.to_thread(
            save_yaml,
            body.yaml,
            config_path=app_state.config_path,
        )
        if result.ok:
            await asyncio.to_thread(app_state.reload_settings)
    return {
        "ok": result.ok,
        "issues": format_issues(result.validation.issues),
        "backup": str(result.backup_path) if result.backup_path else None,
    }


@router.post("/kite/exchange")
async def kite_exchange(request: Request, body: KiteExchangeRequest) -> dict[str, Any]:
    """Exchange a request token and write ``KITE_ACCESS_TOKEN`` to ``.env``."""
    app_state: AppState = request.app.state.dashboard
    service = get_kite_service(app_state)
    async with app_state.write_lock:
        try:
            access_token = await asyncio.to_thread(service.exchange, body.request_token)
            await asyncio.to_thread(app_state.reload_settings)
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
    return {"ok": True, "token_preview": f"{access_token[:6]}…"}


@router.post("/orders/{client_order_id}/mark")
async def orders_mark(
    request: Request,
    client_order_id: str,
    body: OrderMarkRequest,
) -> dict[str, Any]:
    """Mark an OPEN order as FAILED or CANCELLED (operator override)."""
    app_state: AppState = request.app.state.dashboard
    orders = get_orders_service(app_state)
    target_state = OrderState(body.state)
    async with app_state.write_lock:
        try:
            updated = await asyncio.to_thread(
                orders.mark,
                client_order_id,
                state=target_state,
                reason=body.reason,
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"order not found: {client_order_id}",
        )
    return {"ok": True, "client_order_id": client_order_id, "state": updated.state.value}


@router.post("/strategies/{strategy_id}/toggle")
async def strategies_toggle(request: Request, strategy_id: str) -> dict[str, Any]:
    """Flip the persisted enabled flag for ``strategy_id``."""
    app_state: AppState = request.app.state.dashboard
    svc = get_strategy_state(app_state)
    async with app_state.write_lock:
        setting = await asyncio.to_thread(svc.toggle, strategy_id)
    return {
        "strategy_id": setting.strategy_id,
        "enabled": setting.enabled,
        "updated_at": setting.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _overview_snapshot(
    state: AppState,
    kill: KillSwitchService,
    journal: JournalReader,
) -> dict[str, Any]:
    open_records = state.order_store().list_open()
    counts = journal.event_counts_today()
    return {
        "kill_active": kill.is_active(),
        "open_positions": len(open_records),
        "today_realized_pnl": journal.today_realized_pnl(),
        "signals_today": counts.get("signal", 0),
        "vetos_today": counts.get("verdict", 0),
        "orders_today": counts.get("order", 0),
        "kite_configured": state.settings.kite_configured(),
    }


def _order_dict(record: OrderRecord) -> dict[str, Any]:
    return {
        "client_order_id": record.client_order_id,
        "symbol": record.symbol,
        "side": record.side,
        "qty": record.qty,
        "entry_price": record.entry_price,
        "stop_loss": record.stop_loss,
        "target": record.target,
        "state": record.state.value,
        "fill_price": record.fill_price,
        "exit_price": record.exit_price,
        "pnl": record.pnl,
        "signal_ts": record.signal_ts.isoformat(),
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "strategy_id": record.strategy_id,
        "error": record.error,
    }


def _position_dict(record: OrderRecord) -> dict[str, Any]:
    return {
        "symbol": record.symbol,
        "side": "LONG" if record.side == "BUY" else "SHORT",
        "qty": record.qty,
        "entry_price": record.entry_price,
        "stop_loss": record.stop_loss,
        "target": record.target,
        "fill_price": record.fill_price,
        "state": record.state.value,
        "strategy_id": record.strategy_id,
        "opened_at": record.created_at.isoformat(),
    }


def _flatten_kite(state: AppState) -> None:
    """Run :meth:`KiteBroker.flatten_all` in a worker thread.

    The kite client construction is deferred so importing this module
    doesn't require live credentials. Raises :class:`FlattenIncomplete`
    when residual positions remain.
    """
    from data.kite_client import KiteClient

    client = KiteClient.from_settings(state.settings)
    broker = KiteBroker(client, settings=state.settings, state_store=state.order_store())
    broker.flatten_all()
