"""Persist strategy enable/disable toggles in the dashboard DuckDB.

TRADEOFF: We store the flag for v1 but the orchestrator does not yet
consume it — the live loop hard-codes its strategy. The point is to make
the dashboard the *source of truth* so a future enable-check in
``orchestrator/loop.py`` is a one-line read. We document the gap in the
``/strategies`` page so users know toggling is informational today.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb
import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StrategySetting:
    """One persisted strategy-enabled row."""

    strategy_id: str
    enabled: bool
    updated_at: datetime


class StrategyStateService:
    """Read / write strategy enabled flags."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Construct a service bound to the dashboard DuckDB connection."""
        self._conn = conn

    def get(self, strategy_id: str, *, default_enabled: bool = True) -> bool:
        """Return whether ``strategy_id`` is enabled (default if unset)."""
        row = self._conn.execute(
            "SELECT enabled FROM strategy_settings WHERE strategy_id = ?",
            [strategy_id],
        ).fetchone()
        if row is None:
            return default_enabled
        return bool(row[0])

    def set(self, strategy_id: str, *, enabled: bool) -> StrategySetting:
        """Upsert the enabled flag for ``strategy_id``."""
        updated_at = datetime.now().astimezone()
        self._conn.execute(
            "INSERT INTO strategy_settings (strategy_id, enabled, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (strategy_id) DO UPDATE SET "
            "enabled = excluded.enabled, updated_at = excluded.updated_at",
            [strategy_id, enabled, updated_at],
        )
        logger.info(
            "dashboard_strategy_setting_upsert",
            strategy_id=strategy_id,
            enabled=enabled,
        )
        return StrategySetting(
            strategy_id=strategy_id,
            enabled=enabled,
            updated_at=updated_at,
        )

    def list_all(self) -> dict[str, StrategySetting]:
        """Return every persisted strategy setting keyed by id."""
        rows = self._conn.execute(
            "SELECT strategy_id, enabled, updated_at FROM strategy_settings"
        ).fetchall()
        result: dict[str, StrategySetting] = {}
        for row in rows:
            updated_at = row[2]
            if not isinstance(updated_at, datetime):
                updated_at = datetime.fromisoformat(str(updated_at))
            result[str(row[0])] = StrategySetting(
                strategy_id=str(row[0]),
                enabled=bool(row[1]),
                updated_at=updated_at,
            )
        return result

    def toggle(self, strategy_id: str, *, default_enabled: bool = True) -> StrategySetting:
        """Flip the current state of ``strategy_id`` and return the new value."""
        current = self.get(strategy_id, default_enabled=default_enabled)
        return self.set(strategy_id, enabled=not current)
