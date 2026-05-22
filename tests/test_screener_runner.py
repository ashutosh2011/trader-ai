"""End-to-end runner tests with stub provider + in-memory candle store."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pytest

from analyst.providers.mock import MockLLMProvider
from data.store import CandleStore
from screener.llm_screener import DEFAULT_FORMULA, LLMScreener
from screener.prompt import MarketContext
from screener.runner import ScreenerRunner
from screener.store import (
    SCREENER_PICKS_SCHEMA,
    SCREENER_RUNS_SCHEMA,
    ScreenerStore,
)
from screener.universe import Universe, UniverseSymbol

IST = ZoneInfo("Asia/Kolkata")


def _make_bars(count: int, *, base: float, slope: float) -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2024-01-01 09:15:00",
        periods=count,
        freq="1D",
        tz=IST,
    )
    close = base + slope * np.arange(count)
    open_ = np.empty(count)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(count, 1_000.0)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _stub_provider_with_formula(formula_json: str) -> MockLLMProvider:
    return MockLLMProvider(formula_json, name="stub")


def _open_store(tmp_path: Path) -> tuple[duckdb.DuckDBPyConnection, ScreenerStore]:
    conn = duckdb.connect(str(tmp_path / "dash.duckdb"))
    conn.execute(SCREENER_RUNS_SCHEMA)
    conn.execute(SCREENER_PICKS_SCHEMA)
    return conn, ScreenerStore(conn)


def _open_candles(tmp_path: Path) -> CandleStore:
    return CandleStore(tmp_path / "candles.duckdb")


@pytest.mark.asyncio
async def test_runner_end_to_end_persists_picks(tmp_path: Path) -> None:
    conn, store = _open_store(tmp_path)
    candles = _open_candles(tmp_path)
    # PASS symbol — rising close gives close > SMA(50).
    candles.upsert_bars("PASS", "day", _make_bars(120, base=100.0, slope=1.0))
    # FAIL symbol — falling close keeps close < SMA(50).
    candles.upsert_bars("FAIL", "day", _make_bars(120, base=200.0, slope=-1.0))
    universe = Universe(
        symbols=[
            UniverseSymbol(symbol="PASS"),
            UniverseSymbol(symbol="FAIL"),
        ]
    )
    formula_json = (
        '{"name": "above 50 sma", "timeframe": "day", "side_bias": "long", '
        '"rationale": "trending", "filters": ['
        '{"type": "indicator", "indicator": "close", "op": ">", '
        '"compare_to": {"indicator": "sma", "params": {"period": 50}}}'
        "]}"
    )
    llm = LLMScreener(_stub_provider_with_formula(formula_json))
    runner = ScreenerRunner(llm_screener=llm, candle_store=candles, store=store)
    record = await runner.run(
        universe,
        MarketContext(
            as_of=pd.Timestamp.now(tz=IST).to_pydatetime(),
            recent_index_summary="test",
            notes="",
        ),
    )
    try:
        assert record.meta.status == "ok"
        assert len(record.results) == 1
        assert record.results[0].symbol == "PASS"
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].passed_count == 1
        assert runs[0].eligible_count == 2
        assert runs[0].universe_size == 2
        picks = store.list_picks(record.id)
        assert [p.symbol for p in picks] == ["PASS"]
    finally:
        candles.close()
        conn.close()


@pytest.mark.asyncio
async def test_runner_lll_fallback_path_still_persists(tmp_path: Path) -> None:
    conn, store = _open_store(tmp_path)
    candles = _open_candles(tmp_path)
    # Default formula expects bars; populate one symbol that should pass it.
    candles.upsert_bars("OK", "day", _make_bars(120, base=100.0, slope=1.0))
    candles.upsert_bars("BAD", "day", _make_bars(120, base=200.0, slope=-1.0))
    universe = Universe(
        symbols=[
            UniverseSymbol(symbol="OK"),
            UniverseSymbol(symbol="BAD"),
        ]
    )
    # Provider returns garbage → fallback to DEFAULT_FORMULA, which is
    # rsi < 35 AND close > sma(50). Neither symbol is contrived to pass
    # both filters, so we only assert the run was persisted with
    # the fallback status, regardless of pick count.
    llm = LLMScreener(_stub_provider_with_formula("not json"))
    runner = ScreenerRunner(llm_screener=llm, candle_store=candles, store=store)
    record = await runner.run(
        universe,
        MarketContext(
            as_of=pd.Timestamp.now(tz=IST).to_pydatetime(),
            recent_index_summary="test",
            notes="",
        ),
    )
    try:
        assert record.meta.status == "fallback_parse_error"
        assert record.formula == DEFAULT_FORMULA
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].status == "fallback_parse_error"
        assert runs[0].error is not None
    finally:
        candles.close()
        conn.close()


@pytest.mark.asyncio
async def test_runner_skips_symbol_without_candles(tmp_path: Path) -> None:
    conn, store = _open_store(tmp_path)
    candles = _open_candles(tmp_path)
    candles.upsert_bars("ONE", "day", _make_bars(120, base=100.0, slope=1.0))
    universe = Universe(
        symbols=[
            UniverseSymbol(symbol="ONE"),
            UniverseSymbol(symbol="MISSING"),
        ]
    )
    formula_json = (
        '{"name": "above 50 sma", "timeframe": "day", "side_bias": "long", '
        '"rationale": "trending", "filters": ['
        '{"type": "indicator", "indicator": "close", "op": ">", '
        '"compare_to": {"indicator": "sma", "params": {"period": 50}}}'
        "]}"
    )
    llm = LLMScreener(_stub_provider_with_formula(formula_json))
    runner = ScreenerRunner(
        llm_screener=llm,
        candle_store=candles,
        store=store,
        fetcher=None,
    )
    record = await runner.run(
        universe,
        MarketContext(
            as_of=pd.Timestamp.now(tz=IST).to_pydatetime(),
            recent_index_summary="test",
            notes="",
        ),
        fetch_missing=True,  # but no fetcher provided → must not blow up
    )
    try:
        assert record.eligible_count == 1
        assert record.universe_size == 2
        assert [r.symbol for r in record.results] == ["ONE"]
    finally:
        candles.close()
        conn.close()
