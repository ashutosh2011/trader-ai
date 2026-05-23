"""Parameter-sweep runner for the backtest dashboard.

The sweep flow lets the operator pick many symbols × strategy "cells"
(each with its own per-parameter grid) and runs the full cartesian
expansion against Kite historical bars — once per ``(symbol, timeframe,
from_date, to_date)`` — so the leaderboard can rank every result.

TRADEOFF: We run the sweep as a single asyncio task on the event loop
and dispatch each strategy run through :func:`asyncio.to_thread`. This
keeps the request handler latency low (``/api/backtest/sweep/new``
returns immediately) and avoids a heavyweight job queue. The hard cap
on cell count (``MAX_SWEEP_CELLS = 500``) gives a soft upper bound on
how long a single sweep can run.

TRADEOFF: Cross-thread DuckDB access uses a fresh connection per
:func:`asyncio.to_thread` callable because DuckDB connections are not
thread-safe. The sweep status row is updated from the event-loop
thread via :attr:`AppState.dashboard_conn` while each strategy run
opens its own dashboard-DuckDB connection inside the worker thread.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import pandas as pd
import structlog
from kiteconnect.exceptions import KiteException

from config.settings import AppSettings
from dashboard.services.backtest_runner import BacktestRunner
from dashboard.services.instruments import InstrumentsService
from dashboard.services.strategy_schemas import (
    STRATEGY_SCHEMAS,
    ParamSpec,
    strategy_param_keys,
)
from data.historical import HistoricalFetcher
from data.kite_client import KiteClient
from data.store import CandleStore

logger = structlog.get_logger(__name__)

MAX_SWEEP_CELLS = 500

ParamGrid = dict[str, list[Any]]

SweepStatusValue = str  # queued | running | done | failed | cancelled

BarsLoader = Callable[[str, int, str, datetime, datetime], "BarsLoadResult"]
KiteClientFactory = Callable[[AppSettings], KiteClient]


@dataclass(frozen=True)
class BarsLoadResult:
    """Frame + source metadata returned by a sweep bars loader."""

    frame: pd.DataFrame
    source_meta: dict[str, Any]


@dataclass(frozen=True)
class SweepCell:
    """One strategy + per-param grid the sweep iterates over."""

    strategy: str
    param_grid: ParamGrid


@dataclass(frozen=True)
class SweepConfig:
    """Full sweep request submitted by the operator."""

    label: str
    symbols: list[tuple[str, int]]
    cells: list[SweepCell]
    timeframe: str
    from_date: datetime
    to_date: datetime
    qty: int = 1


@dataclass(frozen=True)
class SweepStatus:
    """Snapshot of a sweep row used by the polling endpoint."""

    id: str
    label: str
    status: SweepStatusValue
    total: int
    completed: int
    failed: int
    error: str | None
    elapsed_ms: int
    timeframe: str
    from_date: str
    to_date: str
    qty: int
    created_at: datetime


@dataclass(frozen=True)
class LeaderboardRow:
    """One ranked result row in a finished sweep."""

    rank: int
    run_id: str
    strategy: str
    params: dict[str, Any]
    symbol: str
    total_pnl: float
    sharpe: float
    total_trades: int
    win_rate: float
    mdd_pct: float


@dataclass
class _CellExpansion:
    symbol: str
    instrument_token: int
    strategy: str
    params: dict[str, Any]


@dataclass
class _SweepExecutionState:
    """Mutable in-memory state used while a single sweep runs."""

    sweep_id: str
    config: SweepConfig
    expansions: list[_CellExpansion]
    completed: int = 0
    failed: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)


def expand(
    config: SweepConfig,
) -> list[tuple[str, int, str, dict[str, Any]]]:
    """Return ``(symbol, token, strategy, params)`` cartesian expansions.

    Each cell expands to ``∏(len(values))`` parameter dicts; the full
    sweep size is ``sum_over_cells(cell_expansion) × len(symbols)``.
    Missing param keys mean "use the strategy default" — the persisted
    backtest row records only the explicit overrides.

    Raises:
        ValueError: When the cartesian total exceeds
            ``MAX_SWEEP_CELLS``, when a cell references an unknown
            strategy, when a grid contains an unknown param key, when
            a grid value is empty or outside the param's declared
            ``[min, max]`` range, or when an int-typed param receives a
            non-integral value.
    """
    if not config.symbols:
        msg = "sweep requires at least one symbol"
        raise ValueError(msg)
    if not config.cells:
        msg = "sweep requires at least one strategy cell"
        raise ValueError(msg)

    total_cells_per_symbol = 0
    expanded_cells: list[tuple[str, list[dict[str, Any]]]] = []
    for cell in config.cells:
        if cell.strategy not in STRATEGY_SCHEMAS:
            msg = f"unknown strategy: {cell.strategy}"
            raise ValueError(msg)
        params_list = _expand_cell(cell)
        total_cells_per_symbol += len(params_list)
        expanded_cells.append((cell.strategy, params_list))

    total = total_cells_per_symbol * len(config.symbols)
    if total > MAX_SWEEP_CELLS:
        msg = f"sweep too large: {total} > {MAX_SWEEP_CELLS}"
        raise ValueError(msg)

    out: list[tuple[str, int, str, dict[str, Any]]] = []
    for tradingsymbol, instrument_token in config.symbols:
        for strategy_id, params_list in expanded_cells:
            for params in params_list:
                out.append((tradingsymbol, instrument_token, strategy_id, dict(params)))
    return out


def _expand_cell(cell: SweepCell) -> list[dict[str, Any]]:
    schema = STRATEGY_SCHEMAS[cell.strategy]
    allowed = strategy_param_keys(cell.strategy)
    spec_by_name = {spec.name: spec for spec in schema.params}

    if not cell.param_grid:
        # Empty grid means "use strategy defaults for every param" —
        # exactly one expansion with no overrides.
        return [{}]

    keys: list[str] = []
    value_lists: list[list[Any]] = []
    for name, raw_values in cell.param_grid.items():
        if name not in allowed:
            msg = f"unknown param for {cell.strategy}: {name}"
            raise ValueError(msg)
        if name not in spec_by_name:
            msg = f"unknown param for {cell.strategy}: {name}"
            raise ValueError(msg)
        if not isinstance(raw_values, list) or not raw_values:
            msg = f"param grid for {cell.strategy}.{name} must be a non-empty list"
            raise ValueError(msg)
        spec = spec_by_name[name]
        coerced = [_coerce_param_value(spec, value) for value in raw_values]
        keys.append(name)
        value_lists.append(coerced)

    expansions: list[dict[str, Any]] = [{}]
    for name, values in zip(keys, value_lists, strict=True):
        next_expansions: list[dict[str, Any]] = []
        for partial in expansions:
            for value in values:
                merged = dict(partial)
                merged[name] = value
                next_expansions.append(merged)
        expansions = next_expansions
    return expansions


def _coerce_param_value(spec: ParamSpec, value: object) -> int | float:
    if isinstance(value, bool):
        msg = f"param {spec.name}: bool not allowed"
        raise ValueError(msg)
    if not isinstance(value, int | float):
        msg = f"param {spec.name}: numeric value required (got {type(value).__name__})"
        raise ValueError(msg)
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        msg = f"param {spec.name}: value must be finite"
        raise ValueError(msg)
    if numeric < spec.min or numeric > spec.max:
        msg = (
            f"param {spec.name}: value {value} outside "
            f"[{spec.min}, {spec.max}]"
        )
        raise ValueError(msg)
    if spec.type == "int":
        if isinstance(value, float) and not value.is_integer():
            msg = f"param {spec.name}: integer required (got {value})"
            raise ValueError(msg)
        return int(numeric)
    return float(numeric)


class SweepRunner:
    """Persist + execute parameter sweeps against the dashboard DuckDB."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        settings: AppSettings,
        runner: BacktestRunner,
        instruments: InstrumentsService,
        dashboard_db_path: Path,
        bars_loader: BarsLoader | None = None,
        kite_client_factory: KiteClientFactory | None = None,
    ) -> None:
        """Construct the sweep runner.

        Args:
            conn: Dashboard DuckDB connection used to persist the
                sweep header row and to read status / leaderboard.
            settings: Application settings (Kite credentials live here).
            runner: Backtest runner used by tests that pass a
                pre-built instance; the per-cell worker still opens
                its own connection inside ``asyncio.to_thread``.
            instruments: Cached instruments (validates ``(symbol,
                token)`` pairs at create time).
            dashboard_db_path: Path to the DuckDB file; per-cell
                worker threads open fresh connections at this path so
                they don't share the main connection.
            bars_loader: Override bar fetching (used by tests). The
                default uses :class:`HistoricalFetcher` against Kite.
            kite_client_factory: Override Kite client construction.
        """
        self._conn = conn
        self._settings = settings
        self._runner = runner
        self._instruments = instruments
        self._dashboard_db_path = dashboard_db_path
        self._kite_client_factory = (
            kite_client_factory or KiteClient.from_settings
        )
        self._bars_loader = bars_loader or self._default_bars_loader

    def create(self, config: SweepConfig) -> str:
        """Persist a queued sweep row and return the new sweep id.

        Args:
            config: The vetted sweep configuration. The total cell
                count is derived via :func:`expand` so callers see the
                same validation as the runtime path.

        Raises:
            ValueError: When :func:`expand` rejects the config (cap
                exceeded, unknown strategy / param, or unknown symbol
                token in the cached instruments table).
        """
        for tradingsymbol, instrument_token in config.symbols:
            inst = self._instruments.get_by_token(instrument_token)
            if inst is None or inst.tradingsymbol.upper() != tradingsymbol.upper():
                msg = (
                    f"unknown instrument: tradingsymbol={tradingsymbol} "
                    f"token={instrument_token} (refresh NSE instruments?)"
                )
                raise ValueError(msg)

        expansions = expand(config)
        sweep_id = uuid4().hex[:12]
        created_at = datetime.now().astimezone()
        config_json = json.dumps(_serialize_config(config))
        self._conn.execute(
            "INSERT INTO backtest_sweeps ("
            "id, label, created_at, config_json, timeframe, from_date, to_date, "
            "qty, status, total, completed, failed, error"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                sweep_id,
                config.label or f"sweep · {len(config.symbols)} symbols",
                created_at,
                config_json,
                config.timeframe,
                config.from_date.isoformat(),
                config.to_date.isoformat(),
                int(config.qty),
                "queued",
                int(len(expansions)),
                0,
                0,
                None,
            ],
        )
        logger.info(
            "dashboard_sweep_created",
            sweep_id=sweep_id,
            total=len(expansions),
            symbols=len(config.symbols),
            cells=len(config.cells),
        )
        return sweep_id

    async def run(self, sweep_id: str) -> None:
        """Execute the sweep: fetch bars per symbol, run every expansion."""
        config = self._load_config(sweep_id)
        try:
            expansions_raw = expand(config)
        except ValueError as exc:
            self._mark_failed(sweep_id, str(exc))
            return

        expansions = [
            _CellExpansion(
                symbol=symbol,
                instrument_token=token,
                strategy=strategy_id,
                params=params,
            )
            for symbol, token, strategy_id, params in expansions_raw
        ]
        self._set_status(sweep_id, status="running")

        state = _SweepExecutionState(
            sweep_id=sweep_id,
            config=config,
            expansions=expansions,
        )

        unique_symbols: list[tuple[str, int]] = []
        seen_tokens: set[int] = set()
        for expansion in expansions:
            if expansion.instrument_token in seen_tokens:
                continue
            seen_tokens.add(expansion.instrument_token)
            unique_symbols.append((expansion.symbol, expansion.instrument_token))

        frame_cache: dict[int, BarsLoadResult] = {}
        symbol_load_errors: dict[int, str] = {}
        for tradingsymbol, instrument_token in unique_symbols:
            try:
                load_result = await asyncio.to_thread(
                    self._bars_loader,
                    tradingsymbol,
                    instrument_token,
                    config.timeframe,
                    config.from_date,
                    config.to_date,
                )
            except asyncio.CancelledError:
                self._mark_cancelled(sweep_id, state)
                raise
            except (ValueError, KiteException) as exc:
                symbol_load_errors[instrument_token] = str(exc)
                logger.warning(
                    "dashboard_sweep_bars_load_failed",
                    sweep_id=sweep_id,
                    symbol=tradingsymbol,
                    error=str(exc),
                )
                continue
            frame_cache[instrument_token] = load_result

        for expansion in expansions:
            if expansion.instrument_token not in frame_cache:
                state.failed += 1
                state.failures.append(
                    {
                        "symbol": expansion.symbol,
                        "strategy": expansion.strategy,
                        "params": expansion.params,
                        "error": symbol_load_errors.get(
                            expansion.instrument_token,
                            "bars unavailable",
                        ),
                    }
                )
                self._update_progress(sweep_id, state)
                continue

            try:
                load_result = frame_cache[expansion.instrument_token]
                await asyncio.to_thread(
                    self._run_cell_in_worker,
                    sweep_id,
                    expansion,
                    int(config.qty),
                    load_result,
                )
                state.completed += 1
            except asyncio.CancelledError:
                self._mark_cancelled(sweep_id, state)
                raise
            except Exception as exc:
                state.failed += 1
                state.failures.append(
                    {
                        "symbol": expansion.symbol,
                        "strategy": expansion.strategy,
                        "params": expansion.params,
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "dashboard_sweep_cell_failed",
                    sweep_id=sweep_id,
                    strategy=expansion.strategy,
                    symbol=expansion.symbol,
                    error=str(exc),
                )
            self._update_progress(sweep_id, state)

        final_status = "done" if state.failed < len(expansions) else "failed"
        if state.completed == 0 and state.failed > 0:
            final_status = "failed"
        self._mark_done(
            sweep_id,
            state,
            status=final_status,
            failures=state.failures,
        )
        logger.info(
            "dashboard_sweep_finished",
            sweep_id=sweep_id,
            status=final_status,
            total=len(expansions),
            completed=state.completed,
            failed=state.failed,
        )

    def status(self, sweep_id: str) -> SweepStatus | None:
        """Return the current sweep status row, or ``None`` if missing."""
        row = self._conn.execute(
            "SELECT id, label, created_at, status, total, completed, failed, "
            "error, timeframe, from_date, to_date, qty "
            "FROM backtest_sweeps WHERE id = ?",
            [sweep_id],
        ).fetchone()
        if row is None:
            return None
        created_at = row[2]
        if not isinstance(created_at, datetime):
            created_at = datetime.fromisoformat(str(created_at))
        elapsed_ms = max(
            0,
            int((datetime.now().astimezone() - created_at).total_seconds() * 1000),
        )
        return SweepStatus(
            id=str(row[0]),
            label=str(row[1]),
            status=str(row[3]),
            total=int(row[4]),
            completed=int(row[5]),
            failed=int(row[6]),
            error=str(row[7]) if row[7] is not None else None,
            elapsed_ms=elapsed_ms,
            timeframe=str(row[8]),
            from_date=str(row[9]),
            to_date=str(row[10]),
            qty=int(row[11]),
            created_at=created_at,
        )

    def leaderboard(self, sweep_id: str) -> list[LeaderboardRow]:
        """Return the leaderboard rows for ``sweep_id``."""
        rows = self._conn.execute(
            "SELECT id, strategy, symbol, params, total_pnl, sharpe, "
            "total_trades, win_rate, mdd_pct "
            "FROM backtest_runs WHERE sweep_id = ? "
            "ORDER BY total_pnl DESC, sharpe DESC, id ASC",
            [sweep_id],
        ).fetchall()
        out: list[LeaderboardRow] = []
        for idx, row in enumerate(rows):
            params_payload: dict[str, Any] = json.loads(str(row[3]))
            strategy_params = params_payload.get("strategy", params_payload)
            out.append(
                LeaderboardRow(
                    rank=idx + 1,
                    run_id=str(row[0]),
                    strategy=str(row[1]),
                    symbol=str(row[2]),
                    params=dict(strategy_params)
                    if isinstance(strategy_params, dict)
                    else {},
                    total_pnl=float(row[4]),
                    sharpe=float(row[5]),
                    total_trades=int(row[6]),
                    win_rate=float(row[7]),
                    mdd_pct=float(row[8]),
                )
            )
        return out

    def heatmap(self, sweep_id: str) -> dict[str, Any]:
        """Return the (symbols × strategies) heatmap matrix for the UI."""
        config = self._load_config(sweep_id)
        symbols = [tradingsymbol for tradingsymbol, _ in config.symbols]
        # Preserve the insertion order of cells so duplicate strategies
        # (rare but allowed) line up with the columns the operator picked.
        strategies: list[str] = []
        seen: set[str] = set()
        for cell in config.cells:
            if cell.strategy in seen:
                continue
            seen.add(cell.strategy)
            strategies.append(cell.strategy)
        rows = self._conn.execute(
            "SELECT symbol, strategy, MAX(total_pnl) AS best_pnl "
            "FROM backtest_runs WHERE sweep_id = ? "
            "GROUP BY symbol, strategy",
            [sweep_id],
        ).fetchall()
        best: dict[tuple[str, str], float] = {
            (str(r[0]), str(r[1])): float(r[2]) for r in rows
        }
        cells: list[list[float | None]] = []
        for symbol in symbols:
            row_values: list[float | None] = []
            for strategy_id in strategies:
                row_values.append(best.get((symbol, strategy_id)))
            cells.append(row_values)
        return {"symbols": symbols, "strategies": strategies, "cells": cells}

    def get_config(self, sweep_id: str) -> SweepConfig | None:
        """Return the sweep config or ``None`` when the id is missing."""
        try:
            return self._load_config(sweep_id)
        except KeyError:
            return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_config(self, sweep_id: str) -> SweepConfig:
        row = self._conn.execute(
            "SELECT label, config_json, timeframe, from_date, to_date, qty "
            "FROM backtest_sweeps WHERE id = ?",
            [sweep_id],
        ).fetchone()
        if row is None:
            msg = f"unknown sweep: {sweep_id}"
            raise KeyError(msg)
        payload: dict[str, Any] = json.loads(str(row[1]))
        symbols_raw = payload.get("symbols", [])
        symbols: list[tuple[str, int]] = []
        for entry in symbols_raw:
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            tradingsymbol = str(entry[0])
            instrument_token = int(entry[1])
            symbols.append((tradingsymbol, instrument_token))
        cells_raw = payload.get("cells", [])
        cells: list[SweepCell] = []
        for cell_payload in cells_raw:
            if not isinstance(cell_payload, dict):
                continue
            strategy = str(cell_payload.get("strategy", ""))
            grid_payload = cell_payload.get("param_grid", {}) or {}
            grid: ParamGrid = {}
            if isinstance(grid_payload, dict):
                for key, values in grid_payload.items():
                    if isinstance(values, list):
                        grid[str(key)] = list(values)
            cells.append(SweepCell(strategy=strategy, param_grid=grid))
        return SweepConfig(
            label=str(row[0]),
            symbols=symbols,
            cells=cells,
            timeframe=str(row[2]),
            from_date=_parse_iso(str(row[3])),
            to_date=_parse_iso(str(row[4])),
            qty=int(row[5]),
        )

    def _set_status(self, sweep_id: str, *, status: SweepStatusValue) -> None:
        self._conn.execute(
            "UPDATE backtest_sweeps SET status = ? WHERE id = ?",
            [status, sweep_id],
        )

    def _mark_failed(self, sweep_id: str, error: str) -> None:
        self._conn.execute(
            "UPDATE backtest_sweeps SET status = 'failed', error = ? WHERE id = ?",
            [error, sweep_id],
        )

    def _mark_cancelled(
        self, sweep_id: str, state: _SweepExecutionState
    ) -> None:
        self._conn.execute(
            "UPDATE backtest_sweeps "
            "SET status = 'cancelled', completed = ?, failed = ? WHERE id = ?",
            [int(state.completed), int(state.failed), sweep_id],
        )

    def _update_progress(
        self, sweep_id: str, state: _SweepExecutionState
    ) -> None:
        self._conn.execute(
            "UPDATE backtest_sweeps SET completed = ?, failed = ? WHERE id = ?",
            [int(state.completed), int(state.failed), sweep_id],
        )

    def _mark_done(
        self,
        sweep_id: str,
        state: _SweepExecutionState,
        *,
        status: SweepStatusValue,
        failures: list[dict[str, Any]],
    ) -> None:
        error_summary: str | None = None
        if failures:
            sample = failures[0]
            error_summary = (
                f"{len(failures)} cell(s) failed (e.g. {sample.get('symbol')}/"
                f"{sample.get('strategy')}: {sample.get('error')})"
            )
        self._conn.execute(
            "UPDATE backtest_sweeps "
            "SET status = ?, completed = ?, failed = ?, error = ? "
            "WHERE id = ?",
            [
                status,
                int(state.completed),
                int(state.failed),
                error_summary,
                sweep_id,
            ],
        )

    def _run_cell_in_worker(
        self,
        sweep_id: str,
        expansion: _CellExpansion,
        qty: int,
        load_result: BarsLoadResult,
    ) -> None:
        """Worker callable for :func:`asyncio.to_thread`.

        Opens its own DuckDB connection so the writer is the same
        thread as the connection — a hard requirement DuckDB enforces
        on its non-thread-safe handles.
        """
        worker_conn = duckdb.connect(str(self._dashboard_db_path))
        try:
            runner = BacktestRunner(worker_conn, settings=self._settings)
            runner.run_with_frame(
                strategy_id=expansion.strategy,
                params=expansion.params,
                qty=qty,
                frame=load_result.frame,
                symbol=expansion.symbol,
                source_meta=load_result.source_meta,
                sweep_id=sweep_id,
            )
        finally:
            worker_conn.close()

    def _default_bars_loader(
        self,
        symbol: str,
        instrument_token: int,
        timeframe: str,
        from_date: datetime,
        to_date: datetime,
    ) -> BarsLoadResult:
        """Fetch + cache historical bars via the same path as the runner.

        Runs synchronously inside :func:`asyncio.to_thread` so the
        Kite REST call doesn't stall the event loop.
        """
        if not self._settings.kite_configured():
            msg = (
                "Kite credentials missing — set KITE_API_KEY and refresh "
                "KITE_ACCESS_TOKEN before running a sweep."
            )
            raise ValueError(msg)
        client = self._kite_client_factory(self._settings)
        store = CandleStore(self._settings.data.duckdb_path)
        try:
            try:
                sync = HistoricalFetcher(client, store).fetch_and_store(
                    symbol=symbol,
                    instrument_token=instrument_token,
                    timeframe=timeframe,
                    from_date=from_date,
                    to_date=to_date,
                    fill_gaps=True,
                )
            except KiteException as exc:
                msg = f"Kite historical-data request failed: {exc}"
                raise ValueError(msg) from exc
            frame = store.get_bars(
                symbol, timeframe, start=from_date, end=to_date
            )
        finally:
            store.close()
        if frame.empty:
            msg = (
                f"Kite returned no candles for {symbol} "
                f"token={instrument_token} interval={timeframe}"
            )
            raise ValueError(msg)
        source_meta = {
            "type": "kite",
            "instrument_token": int(instrument_token),
            "timeframe": timeframe,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "rows_fetched": int(sync.rows_fetched),
            "rows_stored": int(sync.rows_stored),
            "gaps_filled": int(sync.gaps_filled),
        }
        return BarsLoadResult(frame=frame, source_meta=source_meta)


def _serialize_config(config: SweepConfig) -> dict[str, Any]:
    return {
        "label": config.label,
        "symbols": [[symbol, int(token)] for symbol, token in config.symbols],
        "cells": [
            {"strategy": cell.strategy, "param_grid": _grid_to_jsonable(cell.param_grid)}
            for cell in config.cells
        ],
        "timeframe": config.timeframe,
        "from_date": config.from_date.isoformat(),
        "to_date": config.to_date.isoformat(),
        "qty": int(config.qty),
    }


def _grid_to_jsonable(grid: ParamGrid) -> dict[str, list[Any]]:
    return {str(k): list(v) for k, v in grid.items()}


def _parse_iso(text: str) -> datetime:
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    return datetime.fromisoformat(cleaned)


__all__ = [
    "MAX_SWEEP_CELLS",
    "BarsLoadResult",
    "BarsLoader",
    "LeaderboardRow",
    "ParamGrid",
    "SweepCell",
    "SweepConfig",
    "SweepRunner",
    "SweepStatus",
    "expand",
]
