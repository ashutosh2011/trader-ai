"""Per-symbol active strategy configuration (applied tuning)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import duckdb
import structlog

logger = structlog.get_logger(__name__)

STRATEGY_SYMBOL_CONFIG_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_symbol_config (
    symbol VARCHAR PRIMARY KEY,
    strategy_id VARCHAR NOT NULL,
    params_json VARCHAR NOT NULL,
    enabled BOOLEAN NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    source_recommendation_id VARCHAR
);
"""


@dataclass(frozen=True)
class SymbolActiveConfig:
    """Active strategy + params for one symbol."""

    symbol: str
    strategy_id: str
    params: dict[str, Any]
    enabled: bool
    updated_at: datetime
    source_recommendation_id: str | None


class ActiveConfigStore:
    """Read / write per-symbol strategy configuration."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self._conn.execute(STRATEGY_SYMBOL_CONFIG_SCHEMA)

    def get(self, symbol: str) -> SymbolActiveConfig | None:
        row = self._conn.execute(
            "SELECT strategy_id, params_json, enabled, updated_at, "
            "source_recommendation_id "
            "FROM strategy_symbol_config WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if row is None:
            return None
        return _row_to_config(symbol, row)

    def list_all(self) -> list[SymbolActiveConfig]:
        rows = self._conn.execute(
            "SELECT symbol, strategy_id, params_json, enabled, updated_at, "
            "source_recommendation_id "
            "FROM strategy_symbol_config ORDER BY symbol"
        ).fetchall()
        return [_row_to_config(str(r[0]), r[1:]) for r in rows]

    def upsert(
        self,
        *,
        symbol: str,
        strategy_id: str,
        params: dict[str, Any],
        enabled: bool,
        source_recommendation_id: str | None = None,
    ) -> SymbolActiveConfig:
        updated_at = datetime.now().astimezone()
        self._conn.execute(
            "INSERT INTO strategy_symbol_config "
            "(symbol, strategy_id, params_json, enabled, updated_at, "
            "source_recommendation_id) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (symbol) DO UPDATE SET "
            "strategy_id = excluded.strategy_id, "
            "params_json = excluded.params_json, "
            "enabled = excluded.enabled, "
            "updated_at = excluded.updated_at, "
            "source_recommendation_id = excluded.source_recommendation_id",
            [
                symbol,
                strategy_id,
                json.dumps(params),
                enabled,
                updated_at,
                source_recommendation_id,
            ],
        )
        logger.info(
            "strategy_symbol_config_upsert",
            symbol=symbol,
            strategy_id=strategy_id,
            enabled=enabled,
        )
        return SymbolActiveConfig(
            symbol=symbol,
            strategy_id=strategy_id,
            params=params,
            enabled=enabled,
            updated_at=updated_at,
            source_recommendation_id=source_recommendation_id,
        )

    def as_lookup(self) -> dict[str, dict[str, Any]]:
        """Return ``symbol -> {strategy_id, params, enabled}`` for collectors."""
        return {
            cfg.symbol: {
                "strategy_id": cfg.strategy_id,
                "params": cfg.params,
                "enabled": cfg.enabled,
            }
            for cfg in self.list_all()
        }


def _row_to_config(symbol: str, row: tuple[Any, ...]) -> SymbolActiveConfig:
    updated_at = row[3]
    if not isinstance(updated_at, datetime):
        updated_at = datetime.fromisoformat(str(updated_at))
    return SymbolActiveConfig(
        symbol=symbol,
        strategy_id=str(row[0]),
        params=json.loads(str(row[1])),
        enabled=bool(row[2]),
        updated_at=updated_at,
        source_recommendation_id=(
            str(row[4]) if row[4] is not None else None
        ),
    )
