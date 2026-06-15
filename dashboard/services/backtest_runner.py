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
from kiteconnect.exceptions import KiteException

from backtest.costs import CostModel
from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import (
    DEFAULT_INITIAL_CAPITAL,
    PerformanceMetrics,
    compute_performance_metrics,
    drawdown_series,
    monthly_returns,
)
from config.settings import AppSettings
from dashboard.services.composite import CombinePolicy, CompositeStrategy
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
    group_id: str | None = None


@dataclass(frozen=True)
class BacktestRunDetail:
    """Full backtest run details for the detail page."""

    summary: BacktestRunSummary
    closed_trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    metrics: dict[str, Any]
    drawdown_curve: list[float]
    monthly_returns: list[dict[str, Any]]
    benchmark_curve: list[dict[str, Any]]
    initial_capital: float


@dataclass(frozen=True)
class BacktestGroupMember:
    """Per-strategy summary inside a comparison group."""

    summary: BacktestRunSummary
    equity_curve: list[dict[str, Any]]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class BacktestGroup:
    """Aggregated header row + per-member summaries for the compare page."""

    id: str
    created_at: datetime
    label: str
    symbol: str
    data_source: str
    bars_count: int
    member_count: int
    source_meta: dict[str, Any]
    members: list[BacktestGroupMember]


@dataclass(frozen=True)
class StrategySelection:
    """One strategy + its params inside a group run request."""

    strategy_id: str
    params: dict[str, Any]


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
        Constructing the runner is idempotent: it ensures the v2
        ``backtest_groups`` table and the ``backtest_runs.group_id``
        column exist so older callers that opened the connection
        directly (e.g. unit-test helpers) keep working.
        """
        self._conn = conn
        self._settings = settings or AppSettings()
        self._kite_client_factory = kite_client_factory or KiteClient.from_settings
        self._ensure_group_schema()

    def _ensure_group_schema(self) -> None:
        """Create v2 group + sweep tables/columns if missing (idempotent)."""
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS backtest_groups ("
            "id VARCHAR PRIMARY KEY,"
            "created_at TIMESTAMPTZ NOT NULL,"
            "label VARCHAR NOT NULL,"
            "symbol VARCHAR NOT NULL,"
            "data_source VARCHAR NOT NULL,"
            "bars_count INTEGER NOT NULL,"
            "member_count INTEGER NOT NULL,"
            "source_meta_json VARCHAR NOT NULL"
            ")"
        )
        for column_sql in (
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS group_id VARCHAR",
            "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS sweep_id VARCHAR",
        ):
            try:
                self._conn.execute(column_sql)
            except duckdb.Error as exc:
                # Older DuckDB versions without ``IF NOT EXISTS`` raise a
                # catalog error when the column already exists; ignore that.
                text = str(exc).lower()
                if "already exists" not in text and "duplicate" not in text:
                    raise

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
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
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
        frame = _tag_symbol(frame, symbol)
        run_id, _ = self._run_member(
            frame=frame,
            symbol=symbol,
            strategy_id=strategy_id,
            params=dict(params or {}),
            qty=qty,
            source_meta=source_meta,
            data_source=data_source,
            group_id=None,
            cost_model=CostModel(commission_pct=commission_pct, slippage_pct=slippage_pct),
        )
        return run_id

    def run_group(
        self,
        *,
        selections: list[StrategySelection],
        symbol: str,
        bars_count: int,
        qty: int = 1,
        seed: int = 42,
        data_source: BacktestDataSource = "synthetic",
        instrument_token: int | None = None,
        timeframe: str | None = None,
        from_date: datetime | str | None = None,
        to_date: datetime | str | None = None,
        label: str | None = None,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> str:
        """Run several strategies against the same bars and persist a group.

        Bars are loaded **once** (synthetic generation or a single Kite
        historical fetch) and reused across every strategy. Each member
        run is persisted via the existing ``backtest_runs`` table with
        the new ``group_id`` column populated; a row in
        ``backtest_groups`` joins them together for the compare page.

        Args:
            selections: Strategies + their per-strategy params.
            symbol: Symbol passed into every strategy and bar frame.
            bars_count: Synthetic bar count (ignored for Kite).
            qty: Default backtest order quantity.
            seed: Synthetic seed.
            data_source: ``"synthetic"`` or ``"kite"``.
            instrument_token: Required for ``"kite"``.
            timeframe: Kite interval, e.g. ``"minute"``.
            from_date: Kite fetch start.
            to_date: Kite fetch end.
            label: Optional display label; defaults to ``"<symbol> · N strategies"``.

        Returns:
            The new group id (UUID4 hex prefix).

        Raises:
            ValueError: For an empty selections list or invalid bar params.
            KeyError: For an unregistered strategy id.
        """
        if not selections:
            msg = "run_group requires at least one strategy selection"
            raise ValueError(msg)
        if bars_count <= 0:
            msg = f"bars_count must be > 0 (got {bars_count})"
            raise ValueError(msg)

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
        frame = _tag_symbol(frame, symbol)

        group_id = uuid4().hex[:12]
        created_at = datetime.now().astimezone()
        cost_model = CostModel(commission_pct=commission_pct, slippage_pct=slippage_pct)
        member_ids: list[str] = []
        for selection in selections:
            run_id, _ = self._run_member(
                # TRADEOFF: Pass a defensive copy so a strategy that
                # accidentally mutates the frame in-place can't poison the
                # next member's run.
                frame=frame.copy(),
                symbol=symbol,
                strategy_id=selection.strategy_id,
                params=dict(selection.params),
                qty=qty,
                source_meta=source_meta,
                data_source=data_source,
                group_id=group_id,
                cost_model=cost_model,
            )
            member_ids.append(run_id)

        resolved_label = label or f"{symbol} · {len(selections)} strategies"
        self._conn.execute(
            "INSERT INTO backtest_groups ("
            "id, created_at, label, symbol, data_source, bars_count, "
            "member_count, source_meta_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                group_id,
                created_at,
                resolved_label,
                symbol,
                str(data_source),
                int(len(frame)),
                len(member_ids),
                json.dumps(source_meta),
            ],
        )
        logger.info(
            "dashboard_backtest_group_persisted",
            group_id=group_id,
            symbol=symbol,
            data_source=data_source,
            member_count=len(member_ids),
        )
        return group_id

    def run_combined(
        self,
        *,
        children: list[StrategySelection],
        policy: CombinePolicy,
        symbol: str,
        bars_count: int,
        qty: int = 1,
        seed: int = 42,
        data_source: BacktestDataSource = "synthetic",
        instrument_token: int | None = None,
        timeframe: str | None = None,
        from_date: datetime | str | None = None,
        to_date: datetime | str | None = None,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> str:
        """Run one composite-strategy backtest and persist a single row.

        Children's per-bar signals are fused into one stream by
        :class:`CompositeStrategy` per ``policy``. Bars are loaded once,
        the composite is wrapped around freshly-constructed child
        instances, and the engine runs against the composite. The
        persisted ``backtest_runs`` row uses ``strategy = "composite"``
        and stashes the per-child setup under ``params["kind"]
        == "composite"`` so the detail page can render the makeup card.

        Args:
            children: Per-child strategy id + params. Duplicate ids are
                allowed (a user may blend two parameterisations of the
                same strategy).
            policy: Direction + price aggregation policy.
            symbol: Symbol passed into every child and bar frame.
            bars_count: Synthetic bar count (ignored for Kite).
            qty: Default backtest order quantity. The composite's per
                bar signal carries ``qty=None`` so the engine applies
                this default uniformly.
            seed: Synthetic seed.
            data_source: ``"synthetic"`` or ``"kite"``.
            instrument_token: Required for ``"kite"``.
            timeframe: Kite interval, e.g. ``"minute"``.
            from_date: Kite fetch start.
            to_date: Kite fetch end.

        Returns:
            The new run id (UUID4 hex prefix).

        Raises:
            ValueError: For an empty / too-large child list, invalid
                bar params, or an unknown child param key.
            KeyError: For an unregistered child strategy id.
        """
        if len(children) < 2:
            msg = "run_combined requires at least 2 children"
            raise ValueError(msg)
        if len(children) > 8:
            msg = "run_combined accepts at most 8 children"
            raise ValueError(msg)
        if bars_count <= 0:
            msg = f"bars_count must be > 0 (got {bars_count})"
            raise ValueError(msg)

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
        frame = _tag_symbol(frame, symbol)

        child_instances: list[Strategy] = []
        sanitized_children: list[dict[str, Any]] = []
        for selection in children:
            strategy_cls = get_strategy(selection.strategy_id)
            allowed = _strategy_param_names(strategy_cls)
            unknown = sorted(set(selection.params) - allowed)
            if unknown:
                msg = (
                    f"unknown params for {selection.strategy_id}: "
                    f"{', '.join(unknown)}"
                )
                raise ValueError(msg)
            clean_params = {k: v for k, v in selection.params.items() if k in allowed}
            child_kwargs: dict[str, Any] = {"symbol": symbol, **clean_params}
            child_instances.append(strategy_cls(**child_kwargs))
            sanitized_children.append(
                {"strategy": selection.strategy_id, "params": clean_params}
            )

        composite = CompositeStrategy(
            children=child_instances,
            policy=policy,
            symbol=symbol,
        )
        cost_model = CostModel(commission_pct=commission_pct, slippage_pct=slippage_pct)
        result = BacktestEngine(qty=qty, cost_model=cost_model).run(composite, frame)
        metrics = compute_performance_metrics(
            result,
            timeframe=_meta_timeframe(source_meta),
            benchmark_prices=_close_prices(frame),
            total_bars=len(frame),
        )
        run_id = uuid4().hex[:12]
        run_at = datetime.now().astimezone()
        stored_params: dict[str, Any] = {
            "kind": "composite",
            "policy": {"direction": policy.direction, "price": policy.price},
            "children": sanitized_children,
            "source": source_meta,
            "costs": {
                "commission_pct": float(commission_pct),
                "slippage_pct": float(slippage_pct),
            },
        }
        self._conn.execute(
            "INSERT INTO backtest_runs ("
            "id, strategy, symbol, params, bars_count, run_at, total_pnl, sharpe, "
            "win_rate, mdd_pct, total_trades, result_json, group_id, sweep_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                "composite",
                symbol,
                json.dumps(stored_params),
                int(len(frame)),
                run_at,
                float(metrics.total_pnl),
                float(metrics.sharpe_ratio),
                float(metrics.win_rate_pct),
                float(metrics.max_drawdown_pct),
                int(metrics.total_trades),
                json.dumps(_serialize_result(result, metrics, frame=frame, qty=qty)),
                None,
                None,
            ],
        )
        logger.info(
            "dashboard_backtest_composite_persisted",
            run_id=run_id,
            symbol=symbol,
            data_source=data_source,
            child_count=len(child_instances),
            direction_policy=policy.direction,
            price_policy=policy.price,
            total_pnl=metrics.total_pnl,
            total_trades=metrics.total_trades,
        )
        return run_id

    def _run_member(
        self,
        *,
        frame: pd.DataFrame,
        symbol: str,
        strategy_id: str,
        params: dict[str, Any],
        qty: int,
        source_meta: dict[str, Any],
        data_source: BacktestDataSource,
        group_id: str | None,
        sweep_id: str | None = None,
        cost_model: CostModel | None = None,
    ) -> tuple[str, PerformanceMetrics]:
        """Run a single strategy against pre-loaded bars and persist it."""
        strategy_cls = get_strategy(strategy_id)
        clean_params = _filter_strategy_kwargs(strategy_cls, params)
        # TRADEOFF: ``Strategy.__init__`` (the base) takes no kwargs, but
        # registered subclasses generally accept ``symbol`` and the params
        # we filtered above. We construct dynamically; mypy can't see the
        # subclass signature so we cast through Any. Unknown kwargs raise
        # ``TypeError`` which the API handler turns into a 400.
        strategy_kwargs: dict[str, Any] = {"symbol": symbol, **clean_params}
        strategy: Strategy = strategy_cls(**strategy_kwargs)

        effective_costs = cost_model or CostModel()
        result = BacktestEngine(qty=qty, cost_model=effective_costs).run(strategy, frame)
        metrics = compute_performance_metrics(
            result,
            timeframe=_meta_timeframe(source_meta),
            benchmark_prices=_close_prices(frame),
            total_bars=len(frame),
        )
        run_id = uuid4().hex[:12]
        run_at = datetime.now().astimezone()
        stored_params = {
            "strategy": clean_params,
            "source": source_meta,
            "costs": {
                "commission_pct": float(effective_costs.commission_pct),
                "slippage_pct": float(effective_costs.slippage_pct),
            },
        }

        self._conn.execute(
            "INSERT INTO backtest_runs ("
            "id, strategy, symbol, params, bars_count, run_at, total_pnl, sharpe, "
            "win_rate, mdd_pct, total_trades, result_json, group_id, sweep_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                json.dumps(_serialize_result(result, metrics, frame=frame, qty=qty)),
                group_id,
                sweep_id,
            ],
        )
        logger.info(
            "dashboard_backtest_persisted",
            run_id=run_id,
            strategy=strategy_id,
            symbol=symbol,
            data_source=data_source,
            group_id=group_id,
            sweep_id=sweep_id,
            total_pnl=metrics.total_pnl,
            total_trades=metrics.total_trades,
        )
        return run_id, metrics

    def run_with_frame(
        self,
        *,
        strategy_id: str,
        params: dict[str, Any],
        qty: int,
        frame: pd.DataFrame,
        symbol: str,
        source_meta: dict[str, Any],
        sweep_id: str | None = None,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> str:
        """Run a backtest against a pre-loaded frame and persist the row.

        Used by :class:`SweepRunner` to amortise a single Kite fetch
        across many ``(strategy, params)`` cells without re-entering
        :meth:`_load_bars`. The persisted row is tagged with
        ``sweep_id`` when supplied; ``group_id`` stays NULL.
        """
        if frame.empty:
            msg = "frame must contain at least one bar"
            raise ValueError(msg)
        tagged = _tag_symbol(frame, symbol)
        run_id, _ = self._run_member(
            frame=tagged,
            symbol=symbol,
            strategy_id=strategy_id,
            params=dict(params),
            qty=qty,
            source_meta=source_meta,
            data_source="kite",
            group_id=None,
            sweep_id=sweep_id,
            cost_model=CostModel(commission_pct=commission_pct, slippage_pct=slippage_pct),
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
        # TRADEOFF: synthetic kept for in-process tests; UI never selects it.
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
            try:
                sync = HistoricalFetcher(client, store).fetch_and_store(
                    symbol=symbol,
                    instrument_token=instrument_token,
                    timeframe=interval,
                    from_date=start,
                    to_date=end,
                    fill_gaps=True,
                )
            except KiteException as exc:
                msg = _kite_historical_error_message(exc)
                raise ValueError(msg) from exc
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

    def list_runs(
        self,
        limit: int = 50,
        *,
        group_id: str | None = None,
    ) -> list[BacktestRunSummary]:
        """Return the ``limit`` most-recent persisted runs (newest first).

        When ``group_id`` is supplied, only runs belonging to that group
        are returned (sorted by ``run_at`` ascending so the compare page
        renders members in execution order).
        """
        if group_id is not None:
            rows = self._conn.execute(
                "SELECT id, strategy, symbol, params, bars_count, run_at, total_pnl, "
                "sharpe, win_rate, mdd_pct, total_trades, group_id "
                "FROM backtest_runs WHERE group_id = ? ORDER BY run_at ASC LIMIT ?",
                [group_id, int(limit)],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, strategy, symbol, params, bars_count, run_at, total_pnl, "
                "sharpe, win_rate, mdd_pct, total_trades, group_id "
                "FROM backtest_runs ORDER BY run_at DESC LIMIT ?",
                [int(limit)],
            ).fetchall()
        return [_row_to_summary(row) for row in rows]

    def get_run(self, run_id: str) -> BacktestRunDetail | None:
        """Return the full detail for ``run_id`` (or ``None`` if missing)."""
        row = self._conn.execute(
            "SELECT id, strategy, symbol, params, bars_count, run_at, total_pnl, "
            "sharpe, win_rate, mdd_pct, total_trades, result_json, group_id "
            "FROM backtest_runs WHERE id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None
        summary = _row_to_summary((*row[:11], row[12]))
        payload: dict[str, Any] = json.loads(str(row[11]))
        return BacktestRunDetail(
            summary=summary,
            closed_trades=list(payload.get("closed_trades", [])),
            equity_curve=list(payload.get("equity_curve", [])),
            metrics=dict(payload.get("metrics", {})),
            drawdown_curve=list(payload.get("drawdown_curve", [])),
            monthly_returns=list(payload.get("monthly_returns", [])),
            benchmark_curve=list(payload.get("benchmark_curve", [])),
            initial_capital=float(payload.get("initial_capital", DEFAULT_INITIAL_CAPITAL)),
        )

    def get_group(self, group_id: str) -> BacktestGroup | None:
        """Return the group header + each member's summary + equity curve.

        Returns ``None`` when ``group_id`` has no row in
        ``backtest_groups``. The equity curves are included so the
        compare page can chart all strategies on the same axis without a
        second round-trip.
        """
        row = self._conn.execute(
            "SELECT id, created_at, label, symbol, data_source, bars_count, "
            "member_count, source_meta_json "
            "FROM backtest_groups WHERE id = ?",
            [group_id],
        ).fetchone()
        if row is None:
            return None
        created_at = row[1]
        if not isinstance(created_at, datetime):
            created_at = datetime.fromisoformat(str(created_at))
        member_rows = self._conn.execute(
            "SELECT id, strategy, symbol, params, bars_count, run_at, total_pnl, "
            "sharpe, win_rate, mdd_pct, total_trades, result_json, group_id "
            "FROM backtest_runs WHERE group_id = ? ORDER BY run_at ASC",
            [group_id],
        ).fetchall()
        members: list[BacktestGroupMember] = []
        for member_row in member_rows:
            summary = _row_to_summary((*member_row[:11], member_row[12]))
            payload: dict[str, Any] = json.loads(str(member_row[11]))
            members.append(
                BacktestGroupMember(
                    summary=summary,
                    equity_curve=list(payload.get("equity_curve", [])),
                    metrics=dict(payload.get("metrics", {})),
                )
            )
        return BacktestGroup(
            id=str(row[0]),
            created_at=created_at,
            label=str(row[2]),
            symbol=str(row[3]),
            data_source=str(row[4]),
            bars_count=int(row[5]),
            member_count=int(row[6]),
            source_meta=json.loads(str(row[7])),
            members=members,
        )


def _filter_strategy_kwargs(
    strategy_cls: type[Strategy],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Filter ``params`` to keys accepted by ``strategy_cls.__init__``.

    Unknown keys are dropped silently so the form can offer a superset
    of params and only the supported ones reach the strategy.
    """
    allowed = _strategy_param_names(strategy_cls)
    return {k: v for k, v in params.items() if k in allowed}


def _strategy_param_names(strategy_cls: type[Strategy]) -> set[str]:
    """Return the kwargs accepted by ``strategy_cls.__init__`` (no ``self``/``symbol``)."""
    sig = inspect.signature(strategy_cls.__init__)
    return {name for name in sig.parameters if name not in {"self", "symbol"}}


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


def _kite_historical_error_message(exc: KiteException) -> str:
    """Build user-facing guidance for Kite historical-data failures."""
    text = str(exc)
    if "api_key" in text or "access_token" in text or "Token" in type(exc).__name__:
        return (
            f"Kite rejected the historical-data request: {text}. "
            "Refresh today's access token from /kite, verify the API key matches "
            "the app that generated the token, then retry."
        )
    return f"Kite historical-data request failed: {text}"


def _row_to_summary(row: tuple[Any, ...]) -> BacktestRunSummary:
    run_at = row[5]
    if not isinstance(run_at, datetime):
        run_at = datetime.fromisoformat(str(run_at))
    group_id: str | None = None
    if len(row) > 11:
        raw = row[11]
        if raw is not None:
            group_id = str(raw)
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
        group_id=group_id,
    )


def _tag_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return ``frame`` with a ``symbol`` column set to ``symbol``."""
    if "symbol" in frame.columns:
        return frame.assign(symbol=symbol)
    out = frame.copy()
    out["symbol"] = symbol
    return out


def _meta_timeframe(source_meta: dict[str, Any]) -> str | None:
    """Pull the Kite interval out of a source-meta dict (None for synthetic)."""
    timeframe = source_meta.get("timeframe")
    return str(timeframe) if timeframe is not None else None


def _close_prices(frame: pd.DataFrame) -> list[float] | None:
    """Return the close-price series for a buy-and-hold benchmark."""
    if "close" not in frame.columns or frame.empty:
        return None
    return [float(v) for v in frame["close"].tolist()]


def _benchmark_curve(frame: pd.DataFrame, qty: int) -> list[dict[str, Any]]:
    """Buy-and-hold PnL per bar (close[i]-close[0])*qty, aligned to bar index.

    Expressed in the same realized-PnL units as the strategy equity curve so
    the two overlay cleanly on the detail chart.
    """
    prices = _close_prices(frame)
    if not prices:
        return []
    base = prices[0]
    return [
        {"bar_index": idx, "equity": (price - base) * qty}
        for idx, price in enumerate(prices)
    ]


def _serialize_result(
    result: BacktestResult,
    metrics: PerformanceMetrics,
    *,
    frame: pd.DataFrame | None = None,
    qty: int = 1,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
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
                "fees": trade.fees,
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
        "drawdown_curve": drawdown_series(
            result.equity_curve, initial_capital=initial_capital
        ),
        "monthly_returns": monthly_returns(
            result.equity_curve, initial_capital=initial_capital
        ),
        "benchmark_curve": (
            _benchmark_curve(frame, qty) if frame is not None else []
        ),
        "initial_capital": initial_capital,
        "metrics": metrics.model_dump(),
    }


__all__ = [
    "BacktestGroup",
    "BacktestGroupMember",
    "BacktestRunDetail",
    "BacktestRunSummary",
    "BacktestRunner",
    "CombinePolicy",
    "Path",
    "StrategySelection",
]
