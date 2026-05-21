"""Run backtests synchronously and persist metrics for the dashboard.

The dashboard's "Run new" form posts strategy + params. We instantiate
the strategy class from :mod:`strategies.registry`, generate synthetic
bars (or accept a future CSV path), run :class:`BacktestEngine`,
compute :class:`PerformanceMetrics`, then store a row in the
``backtest_runs`` DuckDB table with enough JSON to redraw the equity
curve and trade table on the detail page.

TRADEOFF: ``run`` is synchronous — a request to ``/api/backtest/run``
blocks the worker thread until it finishes. Personal-use volumes are
well under a second per run, and the dashboard is a single-user app, so
we accept the latency in exchange for simplicity (no job queue, no
async result polling). Callers from async routes should wrap this in
:func:`asyncio.to_thread`.

TRADEOFF: We treat the EMA-crossover strategy as the only parameterised
strategy in v1 — other strategies can be added by the user, but the
form's ``params`` dict is passed verbatim to the strategy class's
``__init__`` so the strategy must accept keyword args.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import structlog

from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import PerformanceMetrics, compute_performance_metrics
from data.synthetic import make_synthetic_bars
from strategies.base import Strategy
from strategies.registry import get_strategy, list_strategies

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BacktestRunSummary:
    """Persisted summary of a backtest run for the list page."""

    id: str
    strategy: str
    symbol: str
    bars_count: int
    run_at: datetime
    total_pnl: float
    sharpe: float
    win_rate: float
    mdd_pct: float
    total_trades: int
    params: dict[str, Any]


@dataclass(frozen=True)
class BacktestRunDetail:
    """Full backtest run details for the detail page."""

    summary: BacktestRunSummary
    closed_trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    metrics: dict[str, Any]


class BacktestRunner:
    """Run + persist backtests against the dashboard DuckDB table."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Construct a runner bound to a DuckDB connection.

        The connection must already have the ``backtest_runs`` table
        created (see :func:`dashboard.state.AppState.dashboard_conn`).
        """
        self._conn = conn

    def list_strategies(self) -> list[str]:
        """Return registered strategy ids known to the dashboard."""
        return list_strategies()

    def run(
        self,
        *,
        strategy_id: str,
        symbol: str,
        bars_count: int,
        params: dict[str, Any] | None = None,
        qty: int = 1,
        seed: int = 42,
    ) -> str:
        """Run a backtest synchronously and persist the result.

        Args:
            strategy_id: Registered strategy id (e.g. ``"ema_crossover"``).
            symbol: Trading symbol used both in the strategy and bar tag.
            bars_count: Number of synthetic 1-minute bars to generate.
            params: Extra keyword args forwarded to the strategy ``__init__``.
            qty: Default backtest order quantity.
            seed: Synthetic bar seed for reproducible runs.

        Returns:
            The newly-created run id (UUID4 hex prefix).

        Raises:
            KeyError: If ``strategy_id`` is not registered.
            ValueError: If ``bars_count`` is non-positive or ``params``
                contains a key the strategy doesn't accept.
        """
        if bars_count <= 0:
            msg = f"bars_count must be > 0 (got {bars_count})"
            raise ValueError(msg)

        strategy_cls = get_strategy(strategy_id)
        clean_params = _filter_strategy_kwargs(strategy_cls, dict(params or {}))
        # TRADEOFF: ``Strategy.__init__`` (the base) takes no kwargs, but
        # registered subclasses generally accept ``symbol`` and the params
        # we filtered above. We construct dynamically; mypy can't see the
        # subclass signature so we cast through Any. Unknown kwargs raise
        # ``TypeError`` which the API handler turns into a 400.
        strategy_kwargs: dict[str, Any] = {"symbol": symbol, **clean_params}
        strategy: Strategy = strategy_cls(**strategy_kwargs)

        frame = make_synthetic_bars(bars_count, seed=seed)
        if "symbol" in frame.columns:
            frame = frame.assign(symbol=symbol)

        result = BacktestEngine(qty=qty).run(strategy, frame)
        metrics = compute_performance_metrics(result)
        run_id = uuid4().hex[:12]
        run_at = datetime.now().astimezone()

        self._conn.execute(
            "INSERT INTO backtest_runs ("
            "id, strategy, symbol, params, bars_count, run_at, total_pnl, sharpe, "
            "win_rate, mdd_pct, total_trades, result_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                strategy_id,
                symbol,
                json.dumps(clean_params),
                bars_count,
                run_at,
                float(metrics.total_pnl),
                float(metrics.sharpe_ratio),
                float(metrics.win_rate_pct),
                float(metrics.max_drawdown_pct),
                int(metrics.total_trades),
                json.dumps(_serialize_result(result, metrics)),
            ],
        )
        logger.info(
            "dashboard_backtest_persisted",
            run_id=run_id,
            strategy=strategy_id,
            symbol=symbol,
            total_pnl=metrics.total_pnl,
            total_trades=metrics.total_trades,
        )
        return run_id

    def list_runs(self, limit: int = 50) -> list[BacktestRunSummary]:
        """Return the ``limit`` most-recent persisted runs (newest first)."""
        rows = self._conn.execute(
            "SELECT id, strategy, symbol, params, bars_count, run_at, total_pnl, "
            "sharpe, win_rate, mdd_pct, total_trades "
            "FROM backtest_runs ORDER BY run_at DESC LIMIT ?",
            [int(limit)],
        ).fetchall()
        return [_row_to_summary(row) for row in rows]

    def get_run(self, run_id: str) -> BacktestRunDetail | None:
        """Return the full detail for ``run_id`` (or ``None`` if missing)."""
        row = self._conn.execute(
            "SELECT id, strategy, symbol, params, bars_count, run_at, total_pnl, "
            "sharpe, win_rate, mdd_pct, total_trades, result_json "
            "FROM backtest_runs WHERE id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None
        summary = _row_to_summary(row[:11])
        payload: dict[str, Any] = json.loads(str(row[11]))
        return BacktestRunDetail(
            summary=summary,
            closed_trades=list(payload.get("closed_trades", [])),
            equity_curve=list(payload.get("equity_curve", [])),
            metrics=dict(payload.get("metrics", {})),
        )


def _filter_strategy_kwargs(
    strategy_cls: type[Strategy],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Filter ``params`` to keys accepted by ``strategy_cls.__init__``.

    Unknown keys are dropped silently so the form can offer a superset
    of params and only the supported ones reach the strategy.
    """
    sig = inspect.signature(strategy_cls.__init__)
    allowed = {name for name in sig.parameters if name not in {"self", "symbol"}}
    return {k: v for k, v in params.items() if k in allowed}


def _row_to_summary(row: tuple[Any, ...]) -> BacktestRunSummary:
    run_at = row[5]
    if not isinstance(run_at, datetime):
        run_at = datetime.fromisoformat(str(run_at))
    return BacktestRunSummary(
        id=str(row[0]),
        strategy=str(row[1]),
        symbol=str(row[2]),
        params=json.loads(str(row[3])),
        bars_count=int(row[4]),
        run_at=run_at,
        total_pnl=float(row[6]),
        sharpe=float(row[7]),
        win_rate=float(row[8]),
        mdd_pct=float(row[9]),
        total_trades=int(row[10]),
    )


def _serialize_result(
    result: BacktestResult,
    metrics: PerformanceMetrics,
) -> dict[str, Any]:
    """Serialise a :class:`BacktestResult` to a JSON-safe dict."""
    return {
        "closed_trades": [
            {
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "qty": trade.qty,
                "entry_bar": trade.entry_bar,
                "exit_bar": trade.exit_bar,
                "pnl": trade.pnl,
                "exit_reason": trade.exit_reason,
            }
            for trade in result.closed_trades
        ],
        "equity_curve": [
            {
                "bar_index": point.bar_index,
                "timestamp": point.timestamp.isoformat(),
                "equity": point.equity,
            }
            for point in result.equity_curve
        ],
        "metrics": metrics.model_dump(),
    }


__all__ = [
    "BacktestRunDetail",
    "BacktestRunSummary",
    "BacktestRunner",
    "Path",
]
