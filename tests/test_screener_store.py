"""DuckDB roundtrip tests for the screener store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pytest

from screener.llm_screener import DEFAULT_FORMULA, ScreenerMeta
from screener.schema import ScreeningMatch, ScreeningResult
from screener.store import (
    SCREENER_PICKS_SCHEMA,
    SCREENER_RUNS_SCHEMA,
    ScreenerStore,
)

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def store(tmp_path: Path) -> ScreenerStore:
    conn = duckdb.connect(str(tmp_path / "dash.duckdb"))
    conn.execute(SCREENER_RUNS_SCHEMA)
    conn.execute(SCREENER_PICKS_SCHEMA)
    return ScreenerStore(conn)


def _meta(status: str = "ok") -> ScreenerMeta:
    return ScreenerMeta(
        status=status,  # type: ignore[arg-type]
        provider="stub",
        latency_ms=42,
        error=None,
        raw_preview="{}",
    )


def _result(symbol: str) -> ScreeningResult:
    return ScreeningResult(
        symbol=symbol,
        side_bias="long",
        matches=[
            ScreeningMatch(filter_index=0, value=29.5, threshold=30.0, passed=True),
            ScreeningMatch(filter_index=1, value=105.0, threshold=100.0, passed=True),
        ],
        bars_evaluated=120,
        last_bar_ts=datetime(2024, 1, 5, 15, 30, tzinfo=IST),
    )


def test_record_run_and_list_runs(store: ScreenerStore) -> None:
    rid = store.record_run(
        DEFAULT_FORMULA,
        _meta(),
        [_result("AAA"), _result("BBB")],
        universe_size=10,
        eligible_count=8,
    )
    assert rid
    runs = store.list_runs()
    assert len(runs) == 1
    summary = runs[0]
    assert summary.id == rid
    assert summary.universe_size == 10
    assert summary.eligible_count == 8
    assert summary.passed_count == 2
    assert summary.status == "ok"


def test_list_picks_returns_persisted_rows(store: ScreenerStore) -> None:
    rid = store.record_run(
        DEFAULT_FORMULA,
        _meta(),
        [_result("AAA"), _result("BBB")],
        universe_size=5,
        eligible_count=3,
    )
    picks = store.list_picks(rid)
    symbols = [p.symbol for p in picks]
    assert symbols == ["AAA", "BBB"]
    for pick in picks:
        assert pick.bars_evaluated == 120
        assert pick.last_bar_ts.tzinfo is not None
        assert len(pick.matches) == 2


def test_get_run_returns_summary_formula_picks(store: ScreenerStore) -> None:
    rid = store.record_run(
        DEFAULT_FORMULA,
        _meta(),
        [_result("AAA")],
        universe_size=1,
        eligible_count=1,
    )
    detail = store.get_run(rid)
    assert detail is not None
    assert detail.formula == DEFAULT_FORMULA
    assert detail.summary.id == rid
    assert len(detail.picks) == 1


def test_get_run_missing_returns_none(store: ScreenerStore) -> None:
    assert store.get_run("nope") is None


def test_formula_json_round_trips_byte_for_byte(store: ScreenerStore) -> None:
    rid = store.record_run(
        DEFAULT_FORMULA,
        _meta(),
        [],
        universe_size=1,
        eligible_count=0,
    )
    serialized = store.formula_json(rid)
    assert serialized == DEFAULT_FORMULA.model_dump_json()


def test_runs_ordered_by_created_at_desc(store: ScreenerStore) -> None:
    earlier = datetime(2024, 1, 1, 9, 0, tzinfo=IST)
    later = datetime(2024, 1, 2, 9, 0, tzinfo=IST)
    rid_old = store.record_run(
        DEFAULT_FORMULA,
        _meta(),
        [],
        universe_size=1,
        eligible_count=0,
        created_at=earlier,
    )
    rid_new = store.record_run(
        DEFAULT_FORMULA,
        _meta(),
        [],
        universe_size=1,
        eligible_count=0,
        created_at=later,
    )
    runs = store.list_runs()
    assert [r.id for r in runs] == [rid_new, rid_old]


def test_fallback_status_persisted(store: ScreenerStore) -> None:
    rid = store.record_run(
        DEFAULT_FORMULA,
        ScreenerMeta(
            status="fallback_parse_error",
            provider="anthropic",
            latency_ms=120,
            error="boom",
            raw_preview="{}",
        ),
        [],
        universe_size=10,
        eligible_count=10,
    )
    summary = store.list_runs()[0]
    assert summary.id == rid
    assert summary.status == "fallback_parse_error"
    assert summary.error == "boom"
    assert summary.provider == "anthropic"
