"""Lazy singletons shared across dashboard routes and services.

The dashboard wires together a few stateful pieces — :class:`AppSettings`,
the persistent :class:`OrderStateStore`, a DuckDB connection holding
backtest runs and strategy enable/disable flags — and we keep one
instance of each per process. Tests construct a fresh :class:`AppState`
explicitly via :meth:`AppState.build` and override the FastAPI
dependency, so the module-level singleton is only used by the live
server entry-point.

TRADEOFF: We intentionally do **not** lazily import settings at request
time; settings are loaded once at startup so that route handlers see a
stable snapshot. Reloading config is an explicit user action via
``/api/config/save``.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import duckdb
import structlog

from config.settings import AppSettings, load_settings
from execution.order_state import OrderStateStore
from screener.store import SCREENER_PICKS_SCHEMA, SCREENER_RUNS_SCHEMA, ScreenerStore
from tuner.active import STRATEGY_SYMBOL_CONFIG_SCHEMA
from tuner.store import TUNING_RECOMMENDATIONS_SCHEMA, TUNING_RUNS_SCHEMA, TuningStore

logger = structlog.get_logger(__name__)


DASHBOARD_DB_FILENAME = "dashboard.duckdb"

BACKTEST_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    id VARCHAR PRIMARY KEY,
    strategy VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    params VARCHAR NOT NULL,
    bars_count INTEGER NOT NULL,
    run_at TIMESTAMPTZ NOT NULL,
    total_pnl DOUBLE NOT NULL,
    sharpe DOUBLE NOT NULL,
    win_rate DOUBLE NOT NULL,
    mdd_pct DOUBLE NOT NULL,
    total_trades INTEGER NOT NULL,
    result_json VARCHAR NOT NULL
);
"""

# Added in the v2 multi-strategy redesign: every backtest "run group" is one
# row here, and each member run carries a foreign-key-ish ``group_id`` column
# on ``backtest_runs``. Solo runs leave ``group_id`` NULL.
BACKTEST_GROUPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_groups (
    id VARCHAR PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    label VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    data_source VARCHAR NOT NULL,
    bars_count INTEGER NOT NULL,
    member_count INTEGER NOT NULL,
    source_meta_json VARCHAR NOT NULL
);
"""

# TRADEOFF: DuckDB 0.10+ supports ``ADD COLUMN IF NOT EXISTS``; we still
# wrap in try/except so older DuckDB binaries don't blow up the schema
# bootstrap with "duplicate column" errors.
_BACKTEST_RUNS_GROUP_COLUMN = (
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS group_id VARCHAR"
)

STRATEGY_SETTINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_settings (
    strategy_id VARCHAR PRIMARY KEY,
    enabled BOOLEAN NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
"""


class AppState:
    """Container for shared dashboard dependencies.

    The state is constructed once at startup (or per-test) and exposes
    accessors that lazily create heavy resources only when first
    requested. All write operations are serialized through
    :attr:`write_lock` so concurrent HTMX requests from a single user
    can't race each other against the DuckDB file.
    """

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        env_path: Path,
        dashboard_db_path: Path,
        journal_path: Path | None,
    ) -> None:
        self._settings = settings
        self._config_path = config_path
        self._env_path = env_path
        self._dashboard_db_path = dashboard_db_path
        self._journal_path = journal_path
        self._order_store: OrderStateStore | None = None
        self._dashboard_conn: duckdb.DuckDBPyConnection | None = None
        # TRADEOFF: A single async lock serializes mutating endpoints so
        # the dashboard can run on a single-threaded event loop without
        # interleaving config writes / kill toggles. asyncio.Lock is
        # cheap and the dashboard is single-user.
        self.write_lock = asyncio.Lock()
        self._conn_lock = threading.Lock()

    @classmethod
    def build(
        cls,
        *,
        config_path: Path | None = None,
        env_path: Path | None = None,
        dashboard_db_path: Path | None = None,
        journal_path: Path | None = None,
        settings: AppSettings | None = None,
    ) -> AppState:
        """Construct an :class:`AppState` from a config path or settings.

        Args:
            config_path: YAML config file. Defaults to ``config/config.yaml``.
            env_path: ``.env`` file used by the Kite flow. Defaults to ``.env``.
            dashboard_db_path: DuckDB file for dashboard-owned tables.
                Defaults to ``<state_db dir>/dashboard.duckdb``.
            journal_path: Override the journal file shown on ``/journal``.
            settings: Pre-loaded :class:`AppSettings`; when supplied
                ``config_path`` is only used as a metadata hint.

        Returns:
            A ready-to-use :class:`AppState` instance.
        """
        resolved_config = config_path or Path("config/config.yaml")
        resolved_env = env_path or Path(".env")
        loaded = settings if settings is not None else load_settings(
            resolved_config if resolved_config.is_file() else None
        )
        if dashboard_db_path is None:
            dashboard_db_path = loaded.state_db_path.parent / DASHBOARD_DB_FILENAME
        return cls(
            settings=loaded,
            config_path=resolved_config,
            env_path=resolved_env,
            dashboard_db_path=dashboard_db_path,
            journal_path=journal_path,
        )

    @property
    def settings(self) -> AppSettings:
        """Return the currently-loaded :class:`AppSettings`."""
        return self._settings

    @property
    def config_path(self) -> Path:
        """Return the YAML config path used for the editor."""
        return self._config_path

    @property
    def env_path(self) -> Path:
        """Return the ``.env`` path used for the Kite token write."""
        return self._env_path

    @property
    def dashboard_db_path(self) -> Path:
        """Path to the DuckDB file holding dashboard-owned tables."""
        return self._dashboard_db_path

    @property
    def journal_path(self) -> Path | None:
        """Path to the JSONL journal tailed by ``/journal``."""
        return self._journal_path

    def reload_settings(self) -> AppSettings:
        """Re-read settings from disk (used after ``/api/config/save``)."""
        self._settings = load_settings(
            self._config_path if self._config_path.is_file() else None
        )
        return self._settings

    def order_store(self) -> OrderStateStore:
        """Return the singleton :class:`OrderStateStore`."""
        if self._order_store is None:
            self._order_store = OrderStateStore(self._settings.state_db_path)
        return self._order_store

    def dashboard_conn(self) -> duckdb.DuckDBPyConnection:
        """Return the singleton DuckDB connection for dashboard tables."""
        with self._conn_lock:
            if self._dashboard_conn is None:
                self._dashboard_db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = duckdb.connect(str(self._dashboard_db_path))
                conn.execute(BACKTEST_RUNS_SCHEMA)
                conn.execute(BACKTEST_GROUPS_SCHEMA)
                try:
                    conn.execute(_BACKTEST_RUNS_GROUP_COLUMN)
                except duckdb.Error as exc:
                    # Older DuckDB without ``IF NOT EXISTS`` raises a catalog
                    # error when the column already exists; ignore that case.
                    text = str(exc).lower()
                    if "already exists" not in text and "duplicate" not in text:
                        raise
                conn.execute(STRATEGY_SETTINGS_SCHEMA)
                conn.execute(SCREENER_RUNS_SCHEMA)
                conn.execute(SCREENER_PICKS_SCHEMA)
                conn.execute(TUNING_RUNS_SCHEMA)
                conn.execute(TUNING_RECOMMENDATIONS_SCHEMA)
                conn.execute(STRATEGY_SYMBOL_CONFIG_SCHEMA)
                self._dashboard_conn = conn
            return self._dashboard_conn

    def screener_store(self) -> ScreenerStore:
        """Return a :class:`ScreenerStore` bound to the dashboard DuckDB."""
        return ScreenerStore(self.dashboard_conn())

    def tuning_store(self) -> TuningStore:
        """Return a :class:`TuningStore` bound to the dashboard DuckDB."""
        return TuningStore(self.dashboard_conn())

    def close(self) -> None:
        """Release DuckDB handles. Safe to call multiple times."""
        if self._order_store is not None:
            try:
                self._order_store.close()
            except Exception:  # pragma: no cover - close is best-effort
                logger.exception("dashboard_order_store_close_failed")
            self._order_store = None
        if self._dashboard_conn is not None:
            try:
                self._dashboard_conn.close()
            except Exception:  # pragma: no cover - close is best-effort
                logger.exception("dashboard_conn_close_failed")
            self._dashboard_conn = None


_GLOBAL_STATE: AppState | None = None
_GLOBAL_LOCK = threading.Lock()


def get_app_state() -> AppState:
    """Return the process-wide :class:`AppState`, creating it if needed.

    This is the FastAPI dependency used by route handlers. Tests should
    call :func:`set_app_state` with a fixture-built instance instead of
    relying on the global default.
    """
    global _GLOBAL_STATE
    with _GLOBAL_LOCK:
        if _GLOBAL_STATE is None:
            _GLOBAL_STATE = AppState.build()
        return _GLOBAL_STATE


def set_app_state(state: AppState | None) -> None:
    """Replace (or clear) the global app state; intended for tests/CLI."""
    global _GLOBAL_STATE
    with _GLOBAL_LOCK:
        _GLOBAL_STATE = state
