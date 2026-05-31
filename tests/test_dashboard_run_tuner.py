"""Tests for the per-artifact LLM tuner service + endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pytest
from fastapi.testclient import TestClient

from analyst.provider import LLMProvider
from analyst.providers.mock import MockLLMProvider
from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.services.backtest_runner import BacktestRunner
from dashboard.services.instruments import InstrumentsService
from dashboard.services.run_tuner import (
    ParamRecommendation,
    RunTunerService,
    RunTuningPlan,
    SweepTuningPlan,
    build_run_tuning_prompt,
    build_sweep_tuning_prompt,
)
from dashboard.services.strategy_schemas import get_schema
from dashboard.services.sweep_runner import (
    BarsLoadResult,
    SweepCell,
    SweepConfig,
    SweepRunner,
)
from dashboard.state import (
    BACKTEST_GROUPS_SCHEMA,
    BACKTEST_RUNS_SCHEMA,
    BACKTEST_SWEEPS_SCHEMA,
    INSTRUMENTS_META_SCHEMA,
    INSTRUMENTS_SCHEMA,
    RUN_TUNINGS_SCHEMA,
    STRATEGY_SETTINGS_SCHEMA,
    AppState,
    _add_optional_column,
    set_app_state,
)
from data.synthetic import make_synthetic_bars

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


_INSTRUMENTS: list[dict[str, Any]] = [
    {
        "instrument_token": 738561,
        "exchange_token": 0,
        "tradingsymbol": "RELIANCE",
        "name": "Reliance Industries",
        "last_price": 2500.0,
        "expiry": "",
        "strike": 0,
        "tick_size": 0.05,
        "lot_size": 1,
        "instrument_type": "EQ",
        "segment": "NSE",
        "exchange": "NSE",
    },
    {
        "instrument_token": 408065,
        "exchange_token": 0,
        "tradingsymbol": "INFY",
        "name": "Infosys",
        "last_price": 1500.0,
        "expiry": "",
        "strike": 0,
        "tick_size": 0.05,
        "lot_size": 1,
        "instrument_type": "EQ",
        "segment": "NSE",
        "exchange": "NSE",
    },
]


def _instruments_fetcher(_settings: AppSettings, _exchange: str) -> list[dict[str, Any]]:
    return list(_INSTRUMENTS)


def _from_to() -> tuple[datetime, datetime]:
    end = datetime(2024, 1, 5, 15, 30, tzinfo=UTC)
    start = end - timedelta(days=2)
    return start, end


def _synthetic_loader(
    symbol: str,
    instrument_token: int,
    timeframe: str,
    from_date: datetime,
    to_date: datetime,
) -> BarsLoadResult:
    frame = make_synthetic_bars(200, seed=hash(symbol) % 1024)
    meta: dict[str, Any] = {
        "type": "synthetic",
        "instrument_token": int(instrument_token),
        "timeframe": timeframe,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "rows_fetched": 200,
        "rows_stored": 200,
        "gaps_filled": 0,
    }
    return BarsLoadResult(frame=frame, source_meta=meta)


def _build_conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "dash.duckdb"))
    conn.execute(BACKTEST_RUNS_SCHEMA)
    conn.execute(BACKTEST_GROUPS_SCHEMA)
    conn.execute(BACKTEST_SWEEPS_SCHEMA)
    _add_optional_column(
        conn, "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS group_id VARCHAR"
    )
    _add_optional_column(
        conn, "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS sweep_id VARCHAR"
    )
    conn.execute(STRATEGY_SETTINGS_SCHEMA)
    conn.execute(INSTRUMENTS_SCHEMA)
    conn.execute(INSTRUMENTS_META_SCHEMA)
    conn.execute(RUN_TUNINGS_SCHEMA)
    return conn


def _seed_run(conn: duckdb.DuckDBPyConnection) -> str:
    runner = BacktestRunner(conn, settings=AppSettings.model_validate({}))
    run_id = runner.run(
        strategy_id="ema_crossover",
        symbol="SYNTH",
        bars_count=200,
        params={"fast_period": 10, "slow_period": 25},
        qty=1,
        data_source="synthetic",
    )
    return run_id


def _gemini_settings() -> AppSettings:
    return AppSettings.model_validate(
        {"analyst": {"google_api_key": "test-key", "model_google": "gemini-3.5-flash"}}
    )


def _stub_run_plan_json(**overrides: Any) -> str:
    base: dict[str, Any] = {
        "summary": "Drawdown is contained but win rate is low; try a tighter fast EMA.",
        "diagnosis": [
            "fast_period too slow vs slow_period",
            "stops are getting hit before targets",
        ],
        "recommendations": [
            {
                "strategy": "ema_crossover",
                "params": {"fast_period": 8, "slow_period": 21, "atr_period": 14},
                "confidence": 0.7,
                "rationale": "Tighter fast EMA + faster slow EMA on intraday equities.",
            },
            {
                "strategy": "ema_crossover",
                "params": {"fast_period": 5, "slow_period": 13, "atr_period": 10},
                "confidence": 0.45,
                "rationale": "More reactive cross with a shorter ATR.",
            },
        ],
    }
    base.update(overrides)
    return json.dumps(base)


def _stub_sweep_plan_json() -> str:
    payload: dict[str, Any] = {
        "summary": "Top cells cluster around fast=8 / slow=21 — try expanding around it.",
        "leaders": [
            {
                "strategy": "ema_crossover",
                "params": {"fast_period": 8, "slow_period": 21, "atr_period": 14},
                "confidence": 0.8,
                "rationale": "Highest sharpe and PnL on the leaderboard.",
            }
        ],
        "next_sweep": [
            {
                "strategy": "ema_crossover",
                "params": {"fast_period": 7, "slow_period": 19, "atr_period": 14},
                "confidence": 0.55,
                "rationale": "Probe slightly faster around the current leader.",
            }
        ],
        "discard": ["bbands_breakout"],
    }
    return json.dumps(payload)


def _make_service(
    conn: duckdb.DuckDBPyConnection,
    *,
    provider: LLMProvider | None = None,
    settings: AppSettings | None = None,
    dashboard_db_path: Path | None = None,
) -> RunTunerService:
    resolved_settings = settings if settings is not None else _gemini_settings()
    backtest_runner = BacktestRunner(conn, settings=resolved_settings)
    instruments = InstrumentsService(
        conn, settings=resolved_settings, fetcher=_instruments_fetcher
    )
    instruments.ensure_schema()
    instruments.refresh()
    sweep_runner = SweepRunner(
        conn,
        settings=resolved_settings,
        runner=backtest_runner,
        instruments=instruments,
        dashboard_db_path=dashboard_db_path or Path("ignored.duckdb"),
        bars_loader=_synthetic_loader,
    )
    factory = (lambda _s: provider) if provider is not None else None
    service = RunTunerService(
        conn,
        settings=resolved_settings,
        backtest_runner=backtest_runner,
        sweep_runner=sweep_runner,
        provider_factory=factory,
    )
    service.ensure_schema()
    return service


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------


def test_run_tuner_persists_ok_record(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        run_id = _seed_run(conn)
        provider = MockLLMProvider(_stub_run_plan_json(), name="stub-gemini")
        service = _make_service(conn, provider=provider)
        record = asyncio.run(service.tune_run(run_id))

        assert record.status == "ok"
        assert record.scope == "run"
        assert record.target_id == run_id
        assert record.provider == "stub-gemini"
        assert isinstance(record.plan, RunTuningPlan)
        assert len(record.plan.recommendations) == 2
        assert record.plan.recommendations[0].strategy == "ema_crossover"

        row = conn.execute(
            "SELECT id, scope, target_id, status, plan_json "
            "FROM run_tunings WHERE id = ?",
            [record.id],
        ).fetchone()
        assert row is not None
        assert row[1] == "run"
        assert row[2] == run_id
        assert row[3] == "ok"
        plan_payload: dict[str, Any] = json.loads(str(row[4]))
        assert "recommendations" in plan_payload
    finally:
        conn.close()


def test_run_tuner_filters_unknown_keys_and_clamps_bounds(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        run_id = _seed_run(conn)
        # fast_period max is 200; 999 must clamp to 200. ghost_param is
        # not in the schema and must be dropped.
        plan_json = json.dumps(
            {
                "summary": "x",
                "diagnosis": [],
                "recommendations": [
                    {
                        "strategy": "ema_crossover",
                        "params": {
                            "fast_period": 999,
                            "ghost_param": 42,
                            "slow_period": 30,
                        },
                        "confidence": 0.6,
                        "rationale": "clamp test",
                    }
                ],
            }
        )
        provider = MockLLMProvider(plan_json, name="stub")
        service = _make_service(conn, provider=provider)
        record = asyncio.run(service.tune_run(run_id))

        assert record.status == "ok"
        assert isinstance(record.plan, RunTuningPlan)
        rec = record.plan.recommendations[0]
        assert "ghost_param" not in rec.params
        ema_schema = get_schema("ema_crossover")
        assert ema_schema is not None
        fast_spec = next(s for s in ema_schema.params if s.name == "fast_period")
        assert rec.params["fast_period"] == int(fast_spec.max)
        assert rec.params["slow_period"] == 30
    finally:
        conn.close()


class _RaisingTransportProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "stub-transport"

    async def complete(self, prompt: str) -> str:
        raise httpx.HTTPError("simulated transport failure")


def test_run_tuner_fallback_transport(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        run_id = _seed_run(conn)
        service = _make_service(conn, provider=_RaisingTransportProvider())
        record = asyncio.run(service.tune_run(run_id))

        assert record.status == "fallback_transport"
        assert isinstance(record.plan, RunTuningPlan)
        assert record.plan.recommendations == []
        assert record.error is not None
        # Persisted row exists too.
        row = conn.execute(
            "SELECT status FROM run_tunings WHERE id = ?", [record.id]
        ).fetchone()
        assert row is not None and row[0] == "fallback_transport"
    finally:
        conn.close()


def test_run_tuner_fallback_parse_error_keeps_raw_preview(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        run_id = _seed_run(conn)
        provider = MockLLMProvider("definitely not JSON at all", name="garbage")
        service = _make_service(conn, provider=provider)
        record = asyncio.run(service.tune_run(run_id))

        assert record.status == "fallback_parse_error"
        assert record.raw_preview is not None
        assert "not JSON" in record.raw_preview
        assert isinstance(record.plan, RunTuningPlan)
        assert record.plan.recommendations == []
    finally:
        conn.close()


def test_run_tuner_requires_gemini_key_when_no_factory_override(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        run_id = _seed_run(conn)
        # No google_api_key configured AND no factory override → ValueError.
        # We force the analyst block to have no key — pydantic-settings would
        # otherwise overlay the project .env contents during model_validate.
        settings = AppSettings.model_validate({}).model_copy(
            update={
                "google_api_key": None,
                "analyst": AppSettings.model_validate({}).analyst.model_copy(
                    update={"google_api_key": None}
                ),
            }
        )
        service = _make_service(
            conn,
            provider=None,
            settings=settings,
        )
        with pytest.raises(ValueError, match="Gemini API key not configured"):
            asyncio.run(service.tune_run(run_id))
    finally:
        conn.close()


def test_run_tuner_missing_run_raises_keyerror(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        provider = MockLLMProvider(_stub_run_plan_json(), name="stub")
        service = _make_service(conn, provider=provider)
        with pytest.raises(KeyError, match="backtest run not found"):
            asyncio.run(service.tune_run("bogus"))
    finally:
        conn.close()


def test_list_for_run_orders_desc(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        run_id = _seed_run(conn)
        provider = MockLLMProvider(_stub_run_plan_json(), name="stub")
        service = _make_service(conn, provider=provider)
        first = asyncio.run(service.tune_run(run_id))
        time.sleep(0.01)
        second = asyncio.run(service.tune_run(run_id))

        records = service.list_for_run(run_id, limit=5)
        assert len(records) == 2
        assert records[0].id == second.id
        assert records[1].id == first.id

        bounded = service.list_for_run(run_id, limit=1)
        assert len(bounded) == 1
        assert bounded[0].id == second.id
    finally:
        conn.close()


def test_build_run_tuning_prompt_contains_params_and_trades(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        run_id = _seed_run(conn)
        runner = BacktestRunner(conn, settings=AppSettings.model_validate({}))
        detail = runner.get_run(run_id)
        assert detail is not None
        schema = get_schema("ema_crossover")
        assert schema is not None
        prompt = build_run_tuning_prompt(detail, schema)
        assert "ema_crossover" in prompt
        assert "fast_period" in prompt
        assert "Strict JSON" in prompt
        assert "recommendations" in prompt
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Sweep flow
# ---------------------------------------------------------------------------


def _seed_sweep(
    conn: duckdb.DuckDBPyConnection, settings: AppSettings, tmp_path: Path
) -> tuple[str, SweepRunner]:
    instruments = InstrumentsService(
        conn, settings=settings, fetcher=_instruments_fetcher
    )
    instruments.ensure_schema()
    instruments.refresh()
    runner = BacktestRunner(conn, settings=settings)
    sweep_runner = SweepRunner(
        conn,
        settings=settings,
        runner=runner,
        instruments=instruments,
        dashboard_db_path=tmp_path / "dash.duckdb",
        bars_loader=_synthetic_loader,
    )
    start, end = _from_to()
    config = SweepConfig(
        label="ai-sweep",
        symbols=[("RELIANCE", 738561)],
        cells=[
            SweepCell(
                strategy="ema_crossover",
                param_grid={"fast_period": [5, 10]},
            )
        ],
        timeframe="5minute",
        from_date=start,
        to_date=end,
        qty=1,
    )
    sweep_id = sweep_runner.create(config)
    asyncio.run(sweep_runner.run(sweep_id))
    return sweep_id, sweep_runner


def test_sweep_tuner_happy_path(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        settings = _gemini_settings()
        sweep_id, sweep_runner = _seed_sweep(conn, settings, tmp_path)
        provider = MockLLMProvider(_stub_sweep_plan_json(), name="stub")
        service = RunTunerService(
            conn,
            settings=settings,
            backtest_runner=BacktestRunner(conn, settings=settings),
            sweep_runner=sweep_runner,
            provider_factory=lambda _s: provider,
        )
        service.ensure_schema()
        record = asyncio.run(service.tune_sweep(sweep_id))

        assert record.status == "ok"
        assert record.scope == "sweep"
        assert isinstance(record.plan, SweepTuningPlan)
        assert len(record.plan.leaders) == 1
        assert record.plan.leaders[0].strategy == "ema_crossover"
        assert len(record.plan.next_sweep) == 1
        assert record.plan.discard == ["bbands_breakout"]
    finally:
        conn.close()


def test_sweep_tuner_rejects_not_done(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        settings = _gemini_settings()
        instruments = InstrumentsService(
            conn, settings=settings, fetcher=_instruments_fetcher
        )
        instruments.ensure_schema()
        instruments.refresh()
        runner = BacktestRunner(conn, settings=settings)
        sweep_runner = SweepRunner(
            conn,
            settings=settings,
            runner=runner,
            instruments=instruments,
            dashboard_db_path=tmp_path / "dash.duckdb",
            bars_loader=_synthetic_loader,
        )
        start, end = _from_to()
        config = SweepConfig(
            label="queued",
            symbols=[("RELIANCE", 738561)],
            cells=[SweepCell(strategy="ema_crossover", param_grid={"fast_period": [5]})],
            timeframe="5minute",
            from_date=start,
            to_date=end,
            qty=1,
        )
        sweep_id = sweep_runner.create(config)
        # Do NOT run the sweep — it stays in 'queued'.

        provider = MockLLMProvider(_stub_sweep_plan_json(), name="stub")
        service = RunTunerService(
            conn,
            settings=settings,
            backtest_runner=runner,
            sweep_runner=sweep_runner,
            provider_factory=lambda _s: provider,
        )
        service.ensure_schema()
        with pytest.raises(ValueError, match="wait for it to complete"):
            asyncio.run(service.tune_sweep(sweep_id))
    finally:
        conn.close()


def test_build_sweep_prompt_includes_leaderboard_rows(tmp_path: Path) -> None:
    conn = _build_conn(tmp_path)
    try:
        settings = _gemini_settings()
        sweep_id, sweep_runner = _seed_sweep(conn, settings, tmp_path)
        config = sweep_runner.get_config(sweep_id)
        assert config is not None
        leaderboard = sweep_runner.leaderboard(sweep_id)
        schemas = {
            row.strategy: get_schema(row.strategy)
            for row in leaderboard
            if get_schema(row.strategy) is not None
        }
        # mypy: schemas already filtered to non-None
        prompt = build_sweep_tuning_prompt(
            config,
            leaderboard,
            {k: v for k, v in schemas.items() if v is not None},
        )
        assert "Leaderboard" in prompt
        assert "ema_crossover" in prompt
        assert "next_sweep" in prompt
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def api_app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.5\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n", encoding="utf-8")
    settings = AppSettings.model_validate(
        {"analyst": {"google_api_key": "test-key", "model_google": "gemini-3.5-flash"}}
    ).model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "RUN_TUNER_TEST_KILL",
            "state_db_path": tmp_path / "orders.duckdb",
        }
    )
    state = AppState(
        settings=settings,
        config_path=config_path,
        env_path=env_path,
        dashboard_db_path=tmp_path / "dash.duckdb",
        journal_path=None,
    )
    instruments = state.instruments()
    instruments._fetcher = _instruments_fetcher  # noqa: SLF001
    instruments.refresh()
    set_app_state(state)
    try:
        yield state
    finally:
        set_app_state(None)
        state.close()


@pytest.fixture
def api_client(api_app_state: AppState) -> Iterator[TestClient]:
    app = create_app(api_app_state, dev=True)

    import dashboard.routes.api as api_module
    from dashboard.routes._common import get_run_tuner as original_getter

    stub_provider = MockLLMProvider(_stub_run_plan_json(), name="stub-gemini")

    def patched_getter(state: AppState) -> RunTunerService:
        service = original_getter(state)
        service._provider_factory = lambda _s: stub_provider  # noqa: SLF001
        return service

    original_attr = getattr(api_module, "get_run_tuner")  # noqa: B009
    setattr(api_module, "get_run_tuner", patched_getter)  # noqa: B010
    try:
        with TestClient(app) as c:
            yield c
    finally:
        setattr(api_module, "get_run_tuner", original_attr)  # noqa: B010


def _create_run_via_runner(state: AppState) -> str:
    runner = BacktestRunner(state.dashboard_conn(), settings=state.settings)
    return runner.run(
        strategy_id="ema_crossover",
        symbol="SYNTH",
        bars_count=200,
        params={"fast_period": 10, "slow_period": 25},
        qty=1,
        data_source="synthetic",
    )


def test_api_tune_run_ok(api_client: TestClient, api_app_state: AppState) -> None:
    run_id = _create_run_via_runner(api_app_state)
    response = api_client.post(f"/api/backtest/runs/{run_id}/tune")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["scope"] == "run"
    assert body["target_id"] == run_id
    assert len(body["plan"]["recommendations"]) >= 1


def test_api_tune_run_404_for_unknown(api_client: TestClient) -> None:
    response = api_client.post("/api/backtest/runs/bogus/tune")
    assert response.status_code == 404
    assert "backtest run not found" in response.json()["detail"]


def test_api_tunings_list_returns_history(
    api_client: TestClient, api_app_state: AppState
) -> None:
    run_id = _create_run_via_runner(api_app_state)
    first = api_client.post(f"/api/backtest/runs/{run_id}/tune")
    second = api_client.post(f"/api/backtest/runs/{run_id}/tune")
    assert first.status_code == 200
    assert second.status_code == 200
    listing = api_client.get(f"/api/backtest/runs/{run_id}/tunings?limit=5")
    assert listing.status_code == 200
    records = listing.json()["records"]
    assert len(records) == 2
    assert records[0]["created_at"] >= records[1]["created_at"]


def test_api_tune_run_400_when_key_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  max_loss_per_trade_pct: 0.5\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n", encoding="utf-8")
    base = AppSettings.model_validate({})
    settings = base.model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "RUN_TUNER_NOKEY_KILL",
            "state_db_path": tmp_path / "orders.duckdb",
            "google_api_key": None,
            "analyst": base.analyst.model_copy(update={"google_api_key": None}),
        }
    )
    state = AppState(
        settings=settings,
        config_path=config_path,
        env_path=env_path,
        dashboard_db_path=tmp_path / "dash.duckdb",
        journal_path=None,
    )
    instruments = state.instruments()
    instruments._fetcher = _instruments_fetcher  # noqa: SLF001
    instruments.refresh()
    set_app_state(state)
    app = create_app(state, dev=True)
    try:
        with TestClient(app) as client:
            run_id = _create_run_via_runner(state)
            response = client.post(f"/api/backtest/runs/{run_id}/tune")
            assert response.status_code == 400
            assert "Gemini API key" in response.json()["detail"]
    finally:
        set_app_state(None)
        state.close()


def test_api_sweep_tune_400_when_not_done(
    api_client: TestClient, api_app_state: AppState
) -> None:
    # Stand up a queued (never-run) sweep manually so the endpoint hits
    # the "wait for it to complete" guard.
    conn = api_app_state.dashboard_conn()
    runner = BacktestRunner(conn, settings=api_app_state.settings)
    sweep_runner = SweepRunner(
        conn,
        settings=api_app_state.settings,
        runner=runner,
        instruments=api_app_state.instruments(),
        dashboard_db_path=api_app_state.dashboard_db_path,
        bars_loader=_synthetic_loader,
    )
    start, end = _from_to()
    sweep_id = sweep_runner.create(
        SweepConfig(
            label="queued",
            symbols=[("RELIANCE", 738561)],
            cells=[SweepCell(strategy="ema_crossover", param_grid={"fast_period": [5]})],
            timeframe="5minute",
            from_date=start,
            to_date=end,
            qty=1,
        )
    )
    response = api_client.post(f"/api/backtest/sweep/{sweep_id}/tune")
    assert response.status_code == 400
    assert "wait for it to complete" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Page-render smoke tests
# ---------------------------------------------------------------------------


def test_backtest_detail_page_renders_tuning_section(
    api_client: TestClient, api_app_state: AppState
) -> None:
    run_id = _create_run_via_runner(api_app_state)
    response = api_client.get(f"/backtests/{run_id}")
    assert response.status_code == 200, response.text
    assert "AI tuning suggestions" in response.text
    assert "Suggest param tweaks" in response.text


def test_sweep_detail_page_renders_tuning_section(
    api_client: TestClient, api_app_state: AppState
) -> None:
    conn = api_app_state.dashboard_conn()
    runner = BacktestRunner(conn, settings=api_app_state.settings)
    sweep_runner = SweepRunner(
        conn,
        settings=api_app_state.settings,
        runner=runner,
        instruments=api_app_state.instruments(),
        dashboard_db_path=api_app_state.dashboard_db_path,
        bars_loader=_synthetic_loader,
    )
    start, end = _from_to()
    sweep_id = sweep_runner.create(
        SweepConfig(
            label="page-render",
            symbols=[("RELIANCE", 738561)],
            cells=[SweepCell(strategy="ema_crossover", param_grid={"fast_period": [5]})],
            timeframe="5minute",
            from_date=start,
            to_date=end,
            qty=1,
        )
    )
    asyncio.run(sweep_runner.run(sweep_id))

    response = api_client.get(f"/backtests/sweep/{sweep_id}")
    assert response.status_code == 200, response.text
    assert "AI sweep analysis" in response.text
    assert "Analyze leaderboard" in response.text


def test_run_param_recommendation_shape() -> None:
    """Smoke check the shape exported by the service for downstream UI."""
    rec = ParamRecommendation(
        strategy="ema_crossover",
        params={"fast_period": 8},
        confidence=0.9,
        rationale="why",
    )
    assert rec.strategy == "ema_crossover"
    assert rec.confidence == pytest.approx(0.9)
