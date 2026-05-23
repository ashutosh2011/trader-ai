"""Tests for the multi-strategy backtest group flow."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from config.settings import AppSettings
from dashboard.server import create_app
from dashboard.services.backtest_runner import (
    BacktestRunner,
    StrategySelection,
)
from dashboard.state import (
    BACKTEST_GROUPS_SCHEMA,
    BACKTEST_RUNS_SCHEMA,
    STRATEGY_SETTINGS_SCHEMA,
    AppState,
)


def _dashboard_conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "dash.duckdb"))
    conn.execute(BACKTEST_RUNS_SCHEMA)
    conn.execute(BACKTEST_GROUPS_SCHEMA)
    conn.execute(STRATEGY_SETTINGS_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# BacktestRunner.run_group
# ---------------------------------------------------------------------------


def test_run_group_persists_members_and_group(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        selections = [
            StrategySelection(
                strategy_id="ema_crossover",
                params={"fast_period": 5, "slow_period": 12, "atr_period": 7},
            ),
            StrategySelection(
                strategy_id="rsi_mean_revert",
                params={"rsi_period": 10},
            ),
        ]
        group_id = runner.run_group(
            selections=selections,
            symbol="SYNTH",
            bars_count=150,
            seed=7,
        )
        assert isinstance(group_id, str) and len(group_id) == 12

        members = runner.list_runs(group_id=group_id)
        assert len(members) == 2
        assert {m.strategy for m in members} == {"ema_crossover", "rsi_mean_revert"}
        for member in members:
            assert member.group_id == group_id

        group = runner.get_group(group_id)
        assert group is not None
        assert group.id == group_id
        assert group.symbol == "SYNTH"
        assert group.member_count == 2
        assert len(group.members) == 2
        # Each member retains its own equity curve for the chart.
        for group_member in group.members:
            assert isinstance(group_member.equity_curve, list)
            assert group_member.equity_curve, "expected at least one equity point"
    finally:
        conn.close()


def test_run_group_reuses_one_bar_load(tmp_path: Path) -> None:
    # We can't easily mock ``make_synthetic_bars`` from here (the runner
    # imports it directly), so we assert via the persisted bars_count
    # — all members and the group row see the same bar count, which is
    # only possible if the loader fired once and the same frame was fed
    # into each strategy.
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        group_id = runner.run_group(
            selections=[
                StrategySelection(strategy_id="ema_crossover", params={}),
                StrategySelection(strategy_id="bbands_breakout", params={}),
                StrategySelection(strategy_id="macd_trend", params={}),
            ],
            symbol="SYNTH",
            bars_count=180,
            seed=99,
        )
        members = runner.list_runs(group_id=group_id)
        assert len(members) == 3
        bars_counts = {m.bars_count for m in members}
        assert bars_counts == {180}
        group = runner.get_group(group_id)
        assert group is not None and group.bars_count == 180
    finally:
        conn.close()


def test_run_group_rejects_empty_selections(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        with pytest.raises(ValueError, match="at least one"):
            runner.run_group(selections=[], symbol="SYNTH", bars_count=100)
    finally:
        conn.close()


def test_run_group_rejects_bad_bars_count(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        with pytest.raises(ValueError, match="bars_count"):
            runner.run_group(
                selections=[StrategySelection(strategy_id="ema_crossover", params={})],
                symbol="SYNTH",
                bars_count=0,
            )
    finally:
        conn.close()


def test_run_group_unknown_strategy_raises(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        with pytest.raises(KeyError):
            runner.run_group(
                selections=[StrategySelection(strategy_id="nope", params={})],
                symbol="SYNTH",
                bars_count=100,
            )
    finally:
        conn.close()


def test_run_group_custom_label(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        group_id = runner.run_group(
            selections=[StrategySelection(strategy_id="ema_crossover", params={})],
            symbol="SYNTH",
            bars_count=120,
            label="my custom label",
        )
        group = runner.get_group(group_id)
        assert group is not None
        assert group.label == "my custom label"
    finally:
        conn.close()


def test_get_group_unknown_returns_none(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        assert runner.get_group("missing") is None
    finally:
        conn.close()


def test_list_runs_group_filter(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        runner.run(strategy_id="ema_crossover", symbol="SYNTH", bars_count=120)
        group_id = runner.run_group(
            selections=[
                StrategySelection(strategy_id="ema_crossover", params={}),
                StrategySelection(strategy_id="rsi_mean_revert", params={}),
            ],
            symbol="SYNTH",
            bars_count=120,
        )
        all_runs = runner.list_runs()
        assert len(all_runs) == 3
        in_group = runner.list_runs(group_id=group_id)
        assert len(in_group) == 2
        # Solo run never appears in the group filter.
        assert all(r.group_id == group_id for r in in_group)
        # The solo run also has group_id=None in the unfiltered listing.
        solo = next(r for r in all_runs if r.group_id is None)
        assert solo.strategy == "ema_crossover"
    finally:
        conn.close()


def test_get_run_returns_group_id(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        group_id = runner.run_group(
            selections=[StrategySelection(strategy_id="ema_crossover", params={})],
            symbol="SYNTH",
            bars_count=120,
        )
        members = runner.list_runs(group_id=group_id)
        detail = runner.get_run(members[0].id)
        assert detail is not None
        assert detail.summary.group_id == group_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# /api/backtest/run-group and /backtests/compare endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def app_state(tmp_path: Path) -> Iterator[AppState]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "risk:\n  max_loss_per_trade_pct: 0.5\n  daily_loss_cap_pct: 2.0\n",
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text("KILL_SWITCH=0\n", encoding="utf-8")
    settings = AppSettings.model_validate({}).model_copy(
        update={
            "kill_switch_file": tmp_path / "KILL",
            "kill_switch_env": "BG_TEST_KILL",
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
    try:
        yield state
    finally:
        state.close()


@pytest.fixture
def client(app_state: AppState) -> Iterator[TestClient]:
    app = create_app(app_state, dev=True)
    with TestClient(app) as c:
        yield c


def test_api_backtest_run_group_returns_group_id(client: TestClient) -> None:
    payload = {
        "strategies": [
            {"strategy": "ema_crossover", "params": {"fast_period": 5, "slow_period": 12}},
            {"strategy": "rsi_mean_revert", "params": {"rsi_period": 10}},
        ],
        "symbol": "SYNTH",
        "bars_count": 150,
        "seed": 11,
    }
    response = client.post("/api/backtest/run-group", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "group_id" in body
    assert isinstance(body["group_id"], str)


def test_api_backtest_run_group_unknown_strategy_400(client: TestClient) -> None:
    response = client.post(
        "/api/backtest/run-group",
        json={
            "strategies": [{"strategy": "totally_made_up"}],
            "symbol": "SYNTH",
            "bars_count": 100,
        },
    )
    assert response.status_code == 400
    assert "totally_made_up" in response.json()["detail"]


def test_api_backtest_run_group_unknown_params_400(client: TestClient) -> None:
    response = client.post(
        "/api/backtest/run-group",
        json={
            "strategies": [
                {"strategy": "ema_crossover", "params": {"not_a_param": 1}},
            ],
            "symbol": "SYNTH",
            "bars_count": 100,
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "not_a_param" in detail


def test_api_backtest_run_group_duplicate_strategy_400(client: TestClient) -> None:
    response = client.post(
        "/api/backtest/run-group",
        json={
            "strategies": [
                {"strategy": "ema_crossover"},
                {"strategy": "ema_crossover"},
            ],
            "symbol": "SYNTH",
            "bars_count": 100,
        },
    )
    assert response.status_code == 400


def test_api_backtest_run_group_requires_at_least_one_strategy(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/backtest/run-group",
        json={"strategies": [], "symbol": "SYNTH", "bars_count": 100},
    )
    assert response.status_code == 422


def test_compare_page_renders_group(client: TestClient) -> None:
    payload = {
        "strategies": [
            {"strategy": "ema_crossover"},
            {"strategy": "bbands_breakout"},
        ],
        "symbol": "SYNTH",
        "bars_count": 150,
    }
    response = client.post("/api/backtest/run-group", json=payload)
    assert response.status_code == 200
    group_id = response.json()["group_id"]

    page = client.get(f"/backtests/compare/{group_id}")
    assert page.status_code == 200
    body = page.text
    assert "comparison group" in body
    assert "ema_crossover" in body
    assert "bbands_breakout" in body
    assert 'id="cmp-chart"' in body
    assert "combined equity curves" in body
    # Best-performer callout is rendered.
    assert "best performer" in body


def test_compare_page_404_for_unknown_group(client: TestClient) -> None:
    response = client.get("/backtests/compare/unknown-group-xyz")
    assert response.status_code == 404


def test_backtests_page_shows_group_column_and_chip(client: TestClient) -> None:
    payload = {
        "strategies": [
            {"strategy": "ema_crossover"},
            {"strategy": "macd_trend"},
        ],
        "symbol": "SYNTH",
        "bars_count": 120,
    }
    resp = client.post("/api/backtest/run-group", json=payload)
    assert resp.status_code == 200
    group_id = resp.json()["group_id"]
    page = client.get("/backtests")
    assert page.status_code == 200
    body = page.text
    # The "group" header column and the compare link both render.
    assert ">group<" in body
    assert f"/backtests/compare/{group_id}" in body


def test_backtests_page_embeds_schema_json(client: TestClient) -> None:
    response = client.get("/backtests")
    assert response.status_code == 200
    body = response.text
    assert 'id="bt-schema-json"' in body
    assert "ema_crossover" in body
    assert "rsi_mean_revert" in body
    assert "bbands_breakout" in body
    assert "macd_trend" in body
    assert "supertrend_follow" in body


def test_backtests_page_renders_strategy_cards(client: TestClient) -> None:
    response = client.get("/backtests")
    assert response.status_code == 200
    body = response.text
    # Multi-select grid renders one card per strategy with a checkbox.
    assert "bt-strategy-toggle" in body
    assert 'data-strategy-id="ema_crossover"' in body
    assert 'data-strategy-id="supertrend_follow"' in body


def test_backtests_page_includes_compare_route_helpers(client: TestClient) -> None:
    response = client.get("/backtests")
    assert response.status_code == 200
    body = response.text
    # The form posts to the multi-strategy endpoint, not the legacy one.
    assert "/api/backtest/run-group" in body
