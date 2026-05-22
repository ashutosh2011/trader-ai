"""End-to-end screener orchestrator.

A :class:`ScreenerRunner` ties together the LLM screener, the candle
store (optional Kite on-demand fetch), the pure evaluator, and the
persistent :class:`ScreenerStore`. One ``run()`` produces one persisted
record. Designed to be reused from both the CLI and the dashboard.

TRADEOFF: This module performs synchronous I/O (DuckDB, optional Kite
fetch). The dashboard route wraps the synchronous core in
:func:`asyncio.to_thread`. We keep the run shape synchronous so the CLI
and tests don't need an event loop for what is fundamentally a batch
operation, and so the only async boundary in the screener stack is the
LLM provider.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from kiteconnect.exceptions import KiteException

from data.historical import HistoricalFetcher
from data.store import CandleStore
from screener.evaluator import evaluate
from screener.llm_screener import LLMScreener, ScreenerMeta
from screener.prompt import MarketContext
from screener.schema import ScreenerFormula, ScreeningResult
from screener.store import ScreenerStore
from screener.universe import Universe

logger = structlog.get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Map ScreenerFormula timeframe → Kite/CandleStore timeframe.
_KITE_TIMEFRAME: dict[str, str] = {"day": "day", "5m": "5minute"}


@dataclass(frozen=True)
class ScreenerRunRecord:
    """In-memory record returned by :meth:`ScreenerRunner.run`."""

    id: str
    formula: ScreenerFormula
    meta: ScreenerMeta
    results: list[ScreeningResult]
    universe_size: int
    eligible_count: int


class ScreenerRunner:
    """Run one screener pass end-to-end."""

    def __init__(
        self,
        llm_screener: LLMScreener,
        candle_store: CandleStore,
        store: ScreenerStore,
        fetcher: HistoricalFetcher | None = None,
    ) -> None:
        self._llm = llm_screener
        self._candles = candle_store
        self._store = store
        self._fetcher = fetcher

    async def run(
        self,
        universe: Universe,
        market_context: MarketContext,
        *,
        run_id: str | None = None,
        fetch_missing: bool = False,
        bars_back: int = 200,
    ) -> ScreenerRunRecord:
        """Generate the formula, load candles, evaluate, persist, return.

        Args:
            universe: Symbols to consider.
            market_context: Free-form market-regime hints for the LLM.
            run_id: Optional pre-generated id; new UUID prefix when ``None``.
            fetch_missing: If ``True`` AND a :class:`HistoricalFetcher`
                was provided AND the symbol has an ``instrument_token``,
                missing bars are fetched on demand. Kite errors are logged
                and the symbol is skipped — the run never fails.
            bars_back: Lookback window used when fetching/loading bars.

        Returns:
            The persisted :class:`ScreenerRunRecord`.
        """
        rid = run_id or uuid4().hex[:12]
        formula, meta = await self._llm.generate(market_context, universe)
        candles_by_symbol, eligible_count = await asyncio.to_thread(
            self._load_candles,
            universe=universe,
            timeframe=formula.timeframe,
            fetch_missing=fetch_missing,
            bars_back=bars_back,
        )
        results = await asyncio.to_thread(evaluate, formula, candles_by_symbol)
        await asyncio.to_thread(
            self._store.record_run,
            formula,
            meta,
            results,
            universe_size=len(universe.symbols),
            eligible_count=eligible_count,
            run_id=rid,
        )
        return ScreenerRunRecord(
            id=rid,
            formula=formula,
            meta=meta,
            results=results,
            universe_size=len(universe.symbols),
            eligible_count=eligible_count,
        )

    def _load_candles(
        self,
        *,
        universe: Universe,
        timeframe: str,
        fetch_missing: bool,
        bars_back: int,
    ) -> tuple[dict[str, pd.DataFrame], int]:
        store_timeframe = _KITE_TIMEFRAME.get(timeframe, timeframe)
        candles_by_symbol: dict[str, pd.DataFrame] = {}
        eligible_count = 0
        for entry in universe.symbols:
            bars = self._candles.get_bars(entry.symbol, store_timeframe)
            if bars.empty and fetch_missing and entry.instrument_token is not None:
                if self._fetcher is None:
                    logger.debug(
                        "screener_fetch_skipped_no_fetcher",
                        symbol=entry.symbol,
                    )
                else:
                    bars = self._try_fetch(
                        symbol=entry.symbol,
                        instrument_token=entry.instrument_token,
                        timeframe=store_timeframe,
                        bars_back=bars_back,
                    )
            if bars.empty:
                continue
            tail = bars.tail(bars_back).reset_index(drop=True) if bars_back > 0 else bars
            candles_by_symbol[entry.symbol] = tail
            eligible_count += 1
        return candles_by_symbol, eligible_count

    def _try_fetch(
        self,
        *,
        symbol: str,
        instrument_token: int,
        timeframe: str,
        bars_back: int,
    ) -> pd.DataFrame:
        assert self._fetcher is not None  # caller guards
        now = datetime.now(tz=IST)
        # TRADEOFF: a generous lookback window is fine because the Kite
        # historical endpoint caps responses by interval — over-asking
        # just returns whatever is available; under-asking would force
        # second round-trips. We err on the side of pulling enough bars.
        lookback_days = 5 if timeframe == "5minute" else max(bars_back * 2, 90)
        from_date = now - timedelta(days=lookback_days)
        try:
            self._fetcher.fetch_and_store(
                symbol=symbol,
                instrument_token=instrument_token,
                timeframe=timeframe,
                from_date=from_date,
                to_date=now,
                fill_gaps=False,
            )
        except KiteException as exc:
            logger.warning(
                "screener_fetch_kite_error",
                symbol=symbol,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return _empty_ohlcv()
        return self._candles.get_bars(symbol, timeframe)


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


def render_run_record_table(record: ScreenerRunRecord) -> str:
    """Format a run record as a fixed-width table for CLI output."""
    header = (
        f"Run {record.id} — {record.formula.name} "
        f"({record.formula.timeframe} / {record.formula.side_bias})"
    )
    status_line = (
        f"status={record.meta.status} provider={record.meta.provider} "
        f"llm_ms={record.meta.latency_ms} universe={record.universe_size} "
        f"eligible={record.eligible_count} picks={len(record.results)}"
    )
    rationale_line = f"Rationale: {record.formula.rationale}"
    if not record.results:
        return "\n".join([header, status_line, rationale_line, "(no picks)"])
    rows = ["symbol      bars  last_bar_ts                matches"]
    for result in record.results:
        matches_summary = ", ".join(
            f"f{m.filter_index}:{m.value:.2f}vs{m.threshold}" for m in result.matches
        )
        rows.append(
            f"{result.symbol:<10}  {result.bars_evaluated:>5}  "
            f"{result.last_bar_ts.isoformat()}  {matches_summary}"
        )
    return "\n".join([header, status_line, rationale_line, "", *rows])


def render_run_record_json(record: ScreenerRunRecord) -> str:
    """Format a run record as a JSON string for CLI ``--output json``."""
    payload: dict[str, Any] = {
        "id": record.id,
        "formula": record.formula.model_dump(),
        "meta": record.meta.model_dump(),
        "universe_size": record.universe_size,
        "eligible_count": record.eligible_count,
        "results": [
            {
                **result.model_dump(mode="json"),
            }
            for result in record.results
        ],
    }
    return json.dumps(payload, default=str, indent=2)


__all__ = [
    "ScreenerRunRecord",
    "ScreenerRunner",
    "render_run_record_json",
    "render_run_record_table",
]
