"""DuckDB persistence for screener runs and picks.

Mirrors the pattern used by :mod:`dashboard.services.backtest_runner`:
takes a pre-opened DuckDB connection, owns the SQL, returns plain
dataclasses for the routes/templates.

TRADEOFF: ``formula_json`` is stored verbatim (the raw
``ScreenerFormula.model_dump_json()`` string) so the detail page can
re-render the LLM's exact filter shape without re-deriving it from
denormalised columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import duckdb
import structlog

from screener.llm_screener import ScreenerMeta, ScreenerMetaStatus
from screener.schema import ScreenerFormula, ScreeningResult

logger = structlog.get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")

SCREENER_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS screener_runs (
    id VARCHAR PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    timeframe VARCHAR NOT NULL,
    side_bias VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    rationale VARCHAR NOT NULL,
    formula_json VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    provider VARCHAR,
    llm_latency_ms INTEGER,
    universe_size INTEGER NOT NULL,
    eligible_count INTEGER NOT NULL,
    passed_count INTEGER NOT NULL,
    error VARCHAR
);
"""

SCREENER_PICKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS screener_picks (
    run_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    side_bias VARCHAR NOT NULL,
    matches_json VARCHAR NOT NULL,
    bars_evaluated INTEGER NOT NULL,
    last_bar_ts TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (run_id, symbol)
);
"""


@dataclass(frozen=True)
class ScreenerRunSummary:
    """One row in the recent-runs table."""

    id: str
    created_at: datetime
    timeframe: str
    side_bias: str
    name: str
    rationale: str
    status: ScreenerMetaStatus
    provider: str | None
    llm_latency_ms: int | None
    universe_size: int
    eligible_count: int
    passed_count: int
    error: str | None


@dataclass(frozen=True)
class ScreenerPick:
    """One symbol pick within a run."""

    run_id: str
    symbol: str
    side_bias: str
    matches: list[dict[str, Any]]
    bars_evaluated: int
    last_bar_ts: datetime


@dataclass(frozen=True)
class ScreenerRunDetail:
    """Aggregate for the detail page (summary + parsed formula + picks)."""

    summary: ScreenerRunSummary
    formula: ScreenerFormula
    picks: list[ScreenerPick]


class ScreenerStore:
    """Persist + retrieve screener runs and picks from the dashboard DuckDB."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def record_run(
        self,
        formula: ScreenerFormula,
        meta: ScreenerMeta,
        results: list[ScreeningResult],
        *,
        universe_size: int,
        eligible_count: int,
        run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> str:
        """Persist a single run + its picks.

        Args:
            formula: The formula that was evaluated.
            meta: LLM call bookkeeping.
            results: Picks (symbols that passed every filter).
            universe_size: Total symbols in the input universe.
            eligible_count: Universe symbols that had usable candles
                (i.e. passed ``min_bars`` and OHLCV-shape gates).
            run_id: Optional pre-generated id (UUID4 hex prefix); a new
                one is generated when ``None``.
            created_at: Override timestamp; defaults to ``datetime.now()``
                in IST.

        Returns:
            The persisted run id.
        """
        rid = run_id or uuid4().hex[:12]
        ts = (created_at or datetime.now(tz=IST)).astimezone(IST)
        formula_json = formula.model_dump_json()

        self._conn.execute(
            "INSERT INTO screener_runs ("
            "id, created_at, timeframe, side_bias, name, rationale, formula_json, "
            "status, provider, llm_latency_ms, universe_size, eligible_count, "
            "passed_count, error"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rid,
                ts,
                formula.timeframe,
                formula.side_bias,
                formula.name,
                formula.rationale,
                formula_json,
                meta.status,
                meta.provider,
                int(meta.latency_ms),
                int(universe_size),
                int(eligible_count),
                int(len(results)),
                meta.error,
            ],
        )

        for result in results:
            matches_payload = [match.model_dump() for match in result.matches]
            self._conn.execute(
                "INSERT INTO screener_picks ("
                "run_id, symbol, side_bias, matches_json, bars_evaluated, last_bar_ts"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                [
                    rid,
                    result.symbol,
                    result.side_bias,
                    json.dumps(matches_payload),
                    int(result.bars_evaluated),
                    result.last_bar_ts.astimezone(IST),
                ],
            )

        logger.info(
            "screener_run_persisted",
            run_id=rid,
            status=meta.status,
            picks=len(results),
            universe_size=universe_size,
        )
        return rid

    def list_runs(self, limit: int = 20) -> list[ScreenerRunSummary]:
        """Return the most-recent runs, newest first."""
        rows = self._conn.execute(
            "SELECT id, created_at, timeframe, side_bias, name, rationale, "
            "status, provider, llm_latency_ms, universe_size, eligible_count, "
            "passed_count, error "
            "FROM screener_runs ORDER BY created_at DESC LIMIT ?",
            [int(limit)],
        ).fetchall()
        return [_row_to_summary(row) for row in rows]

    def get_run(self, run_id: str) -> ScreenerRunDetail | None:
        """Return summary + parsed formula + picks for ``run_id`` (or None)."""
        row = self._conn.execute(
            "SELECT id, created_at, timeframe, side_bias, name, rationale, "
            "status, provider, llm_latency_ms, universe_size, eligible_count, "
            "passed_count, error, formula_json "
            "FROM screener_runs WHERE id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None
        summary = _row_to_summary(row[:13])
        formula = ScreenerFormula.model_validate_json(str(row[13]))
        picks = self.list_picks(run_id)
        return ScreenerRunDetail(summary=summary, formula=formula, picks=picks)

    def list_picks(self, run_id: str) -> list[ScreenerPick]:
        """Return picks belonging to ``run_id`` (ordered by symbol)."""
        rows = self._conn.execute(
            "SELECT run_id, symbol, side_bias, matches_json, bars_evaluated, last_bar_ts "
            "FROM screener_picks WHERE run_id = ? ORDER BY symbol",
            [run_id],
        ).fetchall()
        picks: list[ScreenerPick] = []
        for row in rows:
            last_ts_raw = row[5]
            last_ts = (
                last_ts_raw
                if isinstance(last_ts_raw, datetime)
                else datetime.fromisoformat(str(last_ts_raw))
            )
            picks.append(
                ScreenerPick(
                    run_id=str(row[0]),
                    symbol=str(row[1]),
                    side_bias=str(row[2]),
                    matches=list(json.loads(str(row[3]))),
                    bars_evaluated=int(row[4]),
                    last_bar_ts=last_ts.astimezone(IST),
                )
            )
        return picks

    def formula_json(self, run_id: str) -> str | None:
        """Return the persisted formula JSON byte string for ``run_id``."""
        row = self._conn.execute(
            "SELECT formula_json FROM screener_runs WHERE id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None
        return str(row[0])


def _row_to_summary(row: tuple[Any, ...]) -> ScreenerRunSummary:
    created_at = row[1]
    if not isinstance(created_at, datetime):
        created_at = datetime.fromisoformat(str(created_at))
    created_at = created_at.astimezone(IST)
    status_raw = str(row[6])
    if status_raw not in {
        "ok",
        "fallback_transport",
        "fallback_parse_error",
        "fallback_unexpected",
    }:
        status_raw = "fallback_unexpected"
    return ScreenerRunSummary(
        id=str(row[0]),
        created_at=created_at,
        timeframe=str(row[2]),
        side_bias=str(row[3]),
        name=str(row[4]),
        rationale=str(row[5]),
        status=status_raw,  # type: ignore[arg-type]
        provider=str(row[7]) if row[7] is not None else None,
        llm_latency_ms=int(row[8]) if row[8] is not None else None,
        universe_size=int(row[9]),
        eligible_count=int(row[10]),
        passed_count=int(row[11]),
        error=str(row[12]) if row[12] is not None else None,
    )


__all__ = [
    "SCREENER_PICKS_SCHEMA",
    "SCREENER_RUNS_SCHEMA",
    "ScreenerPick",
    "ScreenerRunDetail",
    "ScreenerRunSummary",
    "ScreenerStore",
]
