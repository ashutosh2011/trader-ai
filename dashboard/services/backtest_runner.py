"""Run backtests synchronously and persist metrics for the dashboard.

The dashboard's "Run new" form posts strategy + params. We instantiate
the strategy class from :mod:`strategies.registry`, load either synthetic
bars or Kite historical candles, run :class:`BacktestEngine`,
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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import structlog

from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import PerformanceMetrics, compute_performance_metrics
from config.settings import AppSettings
from data.historical import HistoricalFetcher
from data.kite_client import KiteClient
from data.store import CandleStore
from data.synthetic import make_synthetic_bars
from strategies.base import Strategy
from strategies.registry import get_strategy, list_strategies

logger = structlog.get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")

BacktestDataSource = Literal["synthetic", "kite"]


KiteClientFactory = Callable[[AppSettings], KiteClient]


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

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        settings: AppSettings | None = None,
        kite_client_factory: KiteClientFactory | None = None,
    ) -> None:
        """Construct a runner bound to a DuckDB connection.

        The connection must already have the ``backtest_runs`` table
        created (see :func:`dashboard.state.AppState.dashboard_conn`).
        """
        self._conn = conn
        self._settings = settings or AppSettings()
        self._kite_client_factory = kite_client_factory or KiteClient.from_settings

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
        data_source: BacktestDataSource = "synthetic",
        instrument_token: int | None = None,
        timeframe: str | None = None,
        from_date: datetime | str | None = None,
        to_date: datetime | str | None = None,
    ) -> str:
        """Run a backtest synchronously and persist the result.

        Args:
            strategy_id: Registered strategy id (e.g. ``"ema_crossover"``).
            symbol: Trading symbol used both in the strategy and bar tag.
            bars_count: Number of synthetic 1-minute bars to generate. Ignored
                when ``data_source="kite"``; the persisted run stores the
                number of fetched bars instead.
            params: Extra keyword args forwarded to the strategy ``__init__``.
            qty: Default backtest order quantity.
            seed: Synthetic bar seed for reproducible runs.
            data_source: ``"synthetic"`` for generated bars, ``"kite"`` for
                Kite historical candles fetched through the configured account.
            instrument_token: Kite instrument token. Required for
                ``data_source="kite"``.
            timeframe: Kite interval, e.g. ``"minute"`` or ``"5minute"``.
            from_date: Historical fetch start timestamp.
            to_date: Historical fetch end timestamp.

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

        frame, source_meta = self._load_bars(
            data_source=data_source,
            symbol=symbol,
            bars_count=bars_count,
            seed=seed,
            instrument_token=instrument_token,
            timeframe=timeframe,
            from_date=from_date,
            to_date=to_date,
        )
        if "symbol" in frame.columns:
            frame = frame.assign(symbol=symbol)
        else:
            frame["symbol"] = symbol

        result = BacktestEngine(qty=qty).run(strategy, frame)
        metrics = compute_performance_metrics(result)
        run_id = uuid4().hex[:12]
        run_at = datetime.now().astimezone()
        stored_params = {"strategy": clean_params, "source": source_meta}

        self._conn.execute(
            "INSERT INTO backtest_runs ("
            "id, strategy, symbol, params, bars_count, run_at, total_pnl, sharpe, "
            "win_rate, mdd_pct, total_trades, result_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                strategy_id,
                symbol,
                json.dumps(stored_params),
                int(len(frame)),
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
            data_source=data_source,
            total_pnl=metrics.total_pnl,
            total_trades=metrics.total_trades,
        )
        return run_id

    def _load_bars(
        self,
        *,
        data_source: BacktestDataSource,
        symbol: str,
        bars_count: int,
        seed: int,
        instrument_token: int | None,
        timeframe: str | None,
        from_date: datetime | str | None,
        to_date: datetime | str | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if data_source == "synthetic":
            if bars_count <= 0:
                msg = f"bars_count must be > 0 (got {bars_count})"
                raise ValueError(msg)
            frame = make_synthetic_bars(bars_count, seed=seed)
            return frame, {
                "type": "synthetic",
                "bars_count": bars_count,
                "seed": seed,
            }

        if data_source != "kite":
            msg = f"unsupported data_source: {data_source}"
            raise ValueError(msg)
        if not self._settings.kite_configured():
            msg = "Kite API key/access token are required for Kite historical backtests"
            raise ValueError(msg)
        if instrument_token is None or instrument_token <= 0:
            msg = "instrument_token is required for Kite historical backtests"
            raise ValueError(msg)
        interval = timeframe or self._settings.data.default_timeframe
        start = _parse_datetime(from_date, field_name="from_date")
        end = _parse_datetime(to_date, field_name="to_date")
        if end <= start:
            msg = "to_date must be after from_date"
            raise ValueError(msg)

        client = self._kite_client_factory(self._settings)
        store = CandleStore(self._settings.data.duckdb_path)
        try:
            sync = HistoricalFetcher(client, store).fetch_and_store(
                symbol=symbol,
                instrument_token=instrument_token,
                timeframe=interval,
                from_date=start,
                to_date=end,
                fill_gaps=True,
            )
            frame = store.get_bars(symbol, interval, start=start, end=end)
        finally:
            store.close()

        if frame.empty:
            msg = (
                f"Kite returned no candles for {symbol} token={instrument_token} "
                f"interval={interval}"
            )
            raise ValueError(msg)

        return frame, {
            "type": "kite",
            "instrument_token": instrument_token,
            "timeframe": interval,
            "from_date": start.isoformat(),
            "to_date": end.isoformat(),
            "rows_fetched": sync.rows_fetched,
            "rows_stored": sync.rows_stored,
            "gaps_filled": sync.gaps_filled,
        }

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


def _parse_datetime(value: datetime | str | None, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            msg = f"{field_name} must be an ISO datetime"
            raise ValueError(msg) from exc
    else:
        msg = f"{field_name} is required for Kite historical backtests"
        raise ValueError(msg)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


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
