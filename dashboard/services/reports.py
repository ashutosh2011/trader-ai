"""Aggregations over ``backtest_runs`` used by ``/reports`` and ``/overview``.

The dashboard already persists every backtest run; this service answers
"what did I run, and how did it go?" without touching the live trading
loop. Aggregates are computed in SQL (DuckDB) so the queries stay
trivial even with thousands of rows.

TRADEOFF: We deliberately group by ``strategy`` and ``symbol`` only —
not by parameter set — because the cardinality is much friendlier for
a single-screen overview. Drill-down into a specific run uses the
existing ``/backtests/<id>`` detail page.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import duckdb
import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StrategyStats:
    """Per-strategy roll-up for the reports page."""

    strategy: str
    runs: int
    total_trades: int
    total_pnl: float
    avg_pnl: float
    win_rate_pct: float


@dataclass(frozen=True)
class SymbolStats:
    """Per-symbol roll-up for the reports page."""

    symbol: str
    runs: int
    total_trades: int
    total_pnl: float
    avg_pnl: float
    win_rate_pct: float


@dataclass(frozen=True)
class RunHighlight:
    """One row in the "top winners / losers" lists."""

    id: str
    strategy: str
    symbol: str
    total_pnl: float
    run_at: datetime
    total_trades: int


@dataclass(frozen=True)
class DailyPnL:
    """One point on the cumulative-PnL sparkline."""

    day: date
    total_pnl: float
    cumulative_pnl: float


@dataclass(frozen=True)
class OverviewStats:
    """Hero numbers shown at the top of ``/overview``."""

    pnl_7d: float
    pnl_30d: float
    pnl_all_time: float
    total_backtests: int
    total_trades: int
    sparkline: list[DailyPnL]


class ReportsService:
    """Compute report aggregates over the dashboard's DuckDB."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Bind the service to a DuckDB connection (caller-owned)."""
        self._conn = conn

    def by_strategy(self) -> list[StrategyStats]:
        """Return per-strategy aggregates sorted by total_pnl desc."""
        rows = self._conn.execute(
            "SELECT strategy, COUNT(*) AS runs, SUM(total_trades) AS trades, "
            "SUM(total_pnl) AS total_pnl, AVG(total_pnl) AS avg_pnl, "
            "AVG(win_rate) AS win_rate "
            "FROM backtest_runs GROUP BY strategy ORDER BY total_pnl DESC"
        ).fetchall()
        return [
            StrategyStats(
                strategy=str(row[0]),
                runs=int(row[1] or 0),
                total_trades=int(row[2] or 0),
                total_pnl=float(row[3] or 0.0),
                avg_pnl=float(row[4] or 0.0),
                win_rate_pct=float(row[5] or 0.0),
            )
            for row in rows
        ]

    def by_symbol(self) -> list[SymbolStats]:
        """Return per-symbol aggregates sorted by total_pnl desc."""
        rows = self._conn.execute(
            "SELECT symbol, COUNT(*) AS runs, SUM(total_trades) AS trades, "
            "SUM(total_pnl) AS total_pnl, AVG(total_pnl) AS avg_pnl, "
            "AVG(win_rate) AS win_rate "
            "FROM backtest_runs GROUP BY symbol ORDER BY total_pnl DESC"
        ).fetchall()
        return [
            SymbolStats(
                symbol=str(row[0]),
                runs=int(row[1] or 0),
                total_trades=int(row[2] or 0),
                total_pnl=float(row[3] or 0.0),
                avg_pnl=float(row[4] or 0.0),
                win_rate_pct=float(row[5] or 0.0),
            )
            for row in rows
        ]

    def top_winners(self, limit: int = 5) -> list[RunHighlight]:
        """Return the ``limit`` highest-PnL runs (descending)."""
        return self._highlights("DESC", limit)

    def top_losers(self, limit: int = 5) -> list[RunHighlight]:
        """Return the ``limit`` lowest-PnL runs (ascending)."""
        return self._highlights("ASC", limit)

    def _highlights(self, direction: str, limit: int) -> list[RunHighlight]:
        if direction not in {"ASC", "DESC"}:
            msg = f"invalid direction: {direction}"
            raise ValueError(msg)
        rows = self._conn.execute(
            "SELECT id, strategy, symbol, total_pnl, run_at, total_trades "
            f"FROM backtest_runs ORDER BY total_pnl {direction} LIMIT ?",
            [int(limit)],
        ).fetchall()
        return [_row_to_highlight(row) for row in rows]

    def overview_stats(self, *, window_days: int = 30) -> OverviewStats:
        """Compute hero stats + a daily P&L sparkline for the overview page.

        The sparkline groups ``backtest_runs`` by calendar day (in the
        local zone) and emits one point per day in the window, even when
        no runs landed on that day (zero filled).
        """
        now = datetime.now().astimezone()
        today = now.date()
        cutoff_7d = today - timedelta(days=6)
        cutoff_30d = today - timedelta(days=window_days - 1)

        pnl_7d = self._sum_pnl_since(cutoff_7d)
        pnl_30d = self._sum_pnl_since(cutoff_30d)
        all_time = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(total_pnl), 0.0), COALESCE(SUM(total_trades), 0) "
            "FROM backtest_runs"
        ).fetchone()
        total_runs = int(all_time[0] or 0) if all_time else 0
        pnl_all_time = float(all_time[1] or 0.0) if all_time else 0.0
        total_trades = int(all_time[2] or 0) if all_time else 0

        sparkline = self._sparkline(cutoff_30d, today)
        return OverviewStats(
            pnl_7d=pnl_7d,
            pnl_30d=pnl_30d,
            pnl_all_time=pnl_all_time,
            total_backtests=total_runs,
            total_trades=total_trades,
            sparkline=sparkline,
        )

    def _sum_pnl_since(self, cutoff: date) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(total_pnl), 0.0) FROM backtest_runs WHERE run_at >= ?",
            [datetime.combine(cutoff, datetime.min.time()).astimezone()],
        ).fetchone()
        if row is None:
            return 0.0
        return float(row[0] or 0.0)

    def _sparkline(self, start: date, end: date) -> list[DailyPnL]:
        rows = self._conn.execute(
            "SELECT CAST(run_at AS DATE) AS day, SUM(total_pnl) AS pnl "
            "FROM backtest_runs WHERE run_at >= ? "
            "GROUP BY day ORDER BY day",
            [datetime.combine(start, datetime.min.time()).astimezone()],
        ).fetchall()
        by_day: dict[date, float] = {}
        for row in rows:
            row_day = row[0]
            if isinstance(row_day, datetime):
                row_day = row_day.date()
            elif not isinstance(row_day, date):
                row_day = date.fromisoformat(str(row_day))
            by_day[row_day] = float(row[1] or 0.0)
        points: list[DailyPnL] = []
        running = 0.0
        cursor = start
        while cursor <= end:
            day_pnl = by_day.get(cursor, 0.0)
            running += day_pnl
            points.append(DailyPnL(day=cursor, total_pnl=day_pnl, cumulative_pnl=running))
            cursor = cursor + timedelta(days=1)
        return points


def _row_to_highlight(row: tuple[Any, ...]) -> RunHighlight:
    run_at = row[4]
    if not isinstance(run_at, datetime):
        run_at = datetime.fromisoformat(str(run_at))
    return RunHighlight(
        id=str(row[0]),
        strategy=str(row[1]),
        symbol=str(row[2]),
        total_pnl=float(row[3] or 0.0),
        run_at=run_at,
        total_trades=int(row[5] or 0),
    )


__all__ = [
    "DailyPnL",
    "OverviewStats",
    "ReportsService",
    "RunHighlight",
    "StrategyStats",
    "SymbolStats",
]
