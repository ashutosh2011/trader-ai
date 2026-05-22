"""FastAPI app factory and ASGI entry point for the dashboard.

Mounts ``/static``, configures Jinja2 templates with a few cross-page
context helpers (kill-switch banner, active-nav highlighting), and wires
every router under their respective prefixes. The factory accepts an
explicit :class:`AppState` so tests can inject a temp-dir state with a
clean DuckDB file.

TRADEOFF: A single global :class:`AppState` lives on ``app.state.dashboard``.
The factory takes a ``state=`` argument purely for tests — the default
constructs one from ``AppSettings`` at module import time, so the
``uvicorn dashboard.server:app`` entry point works without any setup.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from dashboard.state import AppState

logger = structlog.get_logger(__name__)

DASHBOARD_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"


def create_app(state: AppState | None = None, *, dev: bool = False) -> FastAPI:
    """Build the dashboard FastAPI app.

    Args:
        state: Pre-built :class:`AppState`. When ``None`` the factory
            builds the default one (reads ``config/config.yaml`` and
            opens the order + dashboard DuckDB files lazily on first
            request).
        dev: When ``True`` the error page shows the full traceback. The
            CLI never sets this flag — pass ``--dev`` via env (future
            enhancement) or construct the app from tests.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    app_state = state if state is not None else AppState.build()
    app = FastAPI(
        title="tradebot dashboard",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.dashboard = app_state
    app.state.dev = dev

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals["app_version"] = "0.1.0"
    app.state.templates = templates

    STATIC_DIR.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    _register_middleware(app)
    _register_routers(app)
    _register_error_handlers(app, dev=dev)
    return app


def _register_middleware(app: FastAPI) -> None:
    """Install the request log middleware (method, path, status, ms)."""

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "dashboard_request_error",
                method=request.method,
                path=request.url.path,
            )
            raise
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "dashboard_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response


def _register_routers(app: FastAPI) -> None:
    """Mount every page + API router."""
    # Imports are local to break a potential circular import with
    # dashboard.state (route modules import dashboard.server.templates).
    from dashboard.routes import api, backtests, config_ui, kite_auth, live, overview, strategies
    from dashboard.routes import journal as journal_routes
    from dashboard.routes import orders as orders_routes
    from dashboard.routes import screener as screener_routes

    app.include_router(overview.router)
    app.include_router(live.router)
    app.include_router(orders_routes.router)
    app.include_router(journal_routes.router)
    app.include_router(backtests.router)
    app.include_router(screener_routes.router)
    app.include_router(config_ui.router)
    app.include_router(kite_auth.router)
    app.include_router(strategies.router)
    app.include_router(api.router)


def _register_error_handlers(app: FastAPI, *, dev: bool) -> None:
    """Render a friendly 500 page (with optional traceback in dev mode)."""

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> Response:
        templates: Jinja2Templates = request.app.state.templates
        is_api = request.url.path.startswith("/api/")
        logger.exception(
            "dashboard_unhandled_exception",
            method=request.method,
            path=request.url.path,
            error=str(exc),
        )
        if is_api:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": str(exc) if dev else "see server logs",
                },
            )
        try:
            page = templates.TemplateResponse(
                request,
                "error.html",
                {
                    "title": "500 — internal error",
                    "message": str(exc),
                    "traceback": traceback.format_exc() if dev else None,
                    "kill_banner": _kill_banner_context(request),
                    "now": datetime.now().astimezone(),
                    "active_nav": "",
                },
                status_code=500,
            )
        except Exception:
            return HTMLResponse(
                "<h1>500 — internal error</h1>"
                f"<p>{type(exc).__name__}: {exc!s}</p>",
                status_code=500,
            )
        return page


def get_state(request: Request) -> AppState:
    """FastAPI dependency that returns the :class:`AppState` from app state."""
    state: AppState = request.app.state.dashboard
    return state


def get_templates(request: Request) -> Jinja2Templates:
    """FastAPI dependency that returns the configured Jinja2 templates."""
    templates: Jinja2Templates = request.app.state.templates
    return templates


def _kill_banner_context(request: Request) -> dict[str, Any]:
    """Build the minimal banner context dict used by ``base.html``.

    The function is also called from the error handler so the error
    page renders even if a route handler raised before populating its
    own context.
    """
    state: AppState = request.app.state.dashboard
    try:
        from dashboard.services.kill_switch import KillSwitchService

        kill = KillSwitchService(
            state.settings.kill_switch_file,
            state.settings.kill_switch_env,
        )
        active = kill.is_active()
    except Exception:  # pragma: no cover - defensive
        active = False
    return {"active": active, "kill_file": str(state.settings.kill_switch_file)}


# Module-level app for `uvicorn dashboard.server:app`.
app = create_app()
