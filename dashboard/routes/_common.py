"""Shared route helpers: base template context + service factories.

Every page builds a context with the same kill-banner + nav state, so we
centralise it here. Routes call :func:`base_context` and merge in their
page-specific keys.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates

from dashboard.services.backtest_runner import BacktestRunner
from dashboard.services.config_io import ValidationIssue
from dashboard.services.journal_reader import JournalReader
from dashboard.services.kill_switch import KillSwitchService
from dashboard.services.kite_auth import KiteAuthService
from dashboard.services.orders import OrdersService
from dashboard.services.run_tuner import RunTunerService
from dashboard.services.strategy_state import StrategyStateService
from dashboard.services.sweep_runner import SweepRunner
from dashboard.state import AppState


def get_templates(request: Request) -> Jinja2Templates:
    """Return the Jinja2 templates bound to ``request.app.state``."""
    templates: Jinja2Templates = request.app.state.templates
    return templates


def base_context(request: Request, *, active_nav: str = "") -> dict[str, Any]:
    """Build the cross-page context dict used by ``base.html``.

    Args:
        request: Current FastAPI request (so the template can build URLs).
        active_nav: Identifier of the currently-active nav entry; one of
            ``""`` / ``"overview"`` / ``"live"`` / ``"orders"`` /
            ``"journal"`` / ``"backtests"`` / ``"screener"`` / ``"config"`` /
            ``"kite"`` / ``"strategies"``. Used by ``base.html`` to highlight
            the link.

    Returns:
        A dict that includes ``request`` (Jinja2 requirement on FastAPI)
        plus the kill banner / nav helpers.
    """
    state: AppState = request.app.state.dashboard
    kill_service = KillSwitchService(
        state.settings.kill_switch_file,
        state.settings.kill_switch_env,
    )
    return {
        "request": request,
        "active_nav": active_nav,
        "kill_banner": {
            "active": kill_service.is_active(),
            "kill_file": str(state.settings.kill_switch_file),
        },
        "now": datetime.now().astimezone(),
        "host": request.url.hostname or "127.0.0.1",
        "is_localhost": (request.url.hostname or "127.0.0.1") in {"127.0.0.1", "localhost"},
    }


def get_kill_service(state: AppState) -> KillSwitchService:
    """Construct a :class:`KillSwitchService` for the given app state."""
    return KillSwitchService(
        state.settings.kill_switch_file,
        state.settings.kill_switch_env,
    )


def get_journal_reader(state: AppState) -> JournalReader:
    """Return a :class:`JournalReader` for the dashboard's journal path."""
    return JournalReader(state.journal_path)


def get_orders_service(state: AppState) -> OrdersService:
    """Construct an :class:`OrdersService` from the persistent order store."""
    return OrdersService(state.order_store())


def get_backtest_runner(state: AppState) -> BacktestRunner:
    """Construct a :class:`BacktestRunner` against the dashboard DuckDB."""
    return BacktestRunner(state.dashboard_conn(), settings=state.settings)


def get_sweep_runner(state: AppState) -> SweepRunner:
    """Construct a :class:`SweepRunner` against the dashboard DuckDB."""
    return SweepRunner(
        state.dashboard_conn(),
        settings=state.settings,
        runner=BacktestRunner(state.dashboard_conn(), settings=state.settings),
        instruments=state.instruments(),
        dashboard_db_path=state.dashboard_db_path,
    )


def get_strategy_state(state: AppState) -> StrategyStateService:
    """Construct a :class:`StrategyStateService` against the dashboard DuckDB."""
    return StrategyStateService(state.dashboard_conn())


def get_run_tuner(state: AppState) -> RunTunerService:
    """Return the singleton :class:`RunTunerService` from app state.

    Tests monkey-patch this getter on each route module that imports it
    so the route handler picks up a service with a stubbed provider.
    """
    return state.run_tuner()


def get_kite_service(state: AppState) -> KiteAuthService:
    """Construct a :class:`KiteAuthService` from settings + env path."""
    return KiteAuthService(
        env_path=state.env_path,
        api_key=state.settings.kite.api_key,
        api_secret=state.settings.kite.api_secret,
        access_token=state.settings.kite.access_token,
    )


def format_issues(issues: list[ValidationIssue]) -> list[dict[str, str]]:
    """Render validation issues as plain dicts for templates / JSON."""
    return [
        {"location": issue.location, "message": issue.message, "type": issue.type}
        for issue in issues
    ]
