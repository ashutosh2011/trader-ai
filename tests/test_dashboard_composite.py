"""Tests for CompositeStrategy + the /api/backtest/run-combined flow."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.settings import AppSettings
from core.context import Context
from core.signal import Signal
from dashboard.server import create_app
from dashboard.services.backtest_runner import (
    BacktestRunner,
    StrategySelection,
)
from dashboard.services.composite import (
    CombinePolicy,
    CompositeStrategy,
    aggregate_stop,
    aggregate_target,
    resolve_direction,
)
from dashboard.state import (
    BACKTEST_GROUPS_SCHEMA,
    BACKTEST_RUNS_SCHEMA,
    STRATEGY_SETTINGS_SCHEMA,
    AppState,
)
from indicators.base import Indicator
from strategies.base import Strategy

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Fake child strategy that emits scripted signals per bar index
# ---------------------------------------------------------------------------


@dataclass
class FakeScript:
    """Scripted output for a fake child strategy across a sequence of bars.

    ``signals_by_bar`` maps ``bar_index`` to an optional pre-built
    :class:`Signal`. Missing entries mean "no signal this bar"
    (abstention). The same dict is reused across multiple instances so
    tests can build several children sharing the same id.
    """

    signals_by_bar: dict[int, Signal | None] = field(default_factory=dict)


def _make_fake_child_class(child_id: str) -> type[Strategy]:
    """Build a FakeChild subclass with the given ``id`` class variable.

    Each fake child needs its own ``id`` because :class:`CompositeStrategy`
    reads it from the class to build display tags. Strategy treats
    ``id`` as a ``ClassVar``, so we synthesise a fresh subclass per
    distinct child id rather than shadow the ClassVar on instances.
    """

    class _Fake(Strategy):
        id: ClassVar[str] = child_id
        timeframe: ClassVar[str] = "1m"

        def __init__(self, *, script: FakeScript) -> None:
            super().__init__()
            self._script = script
            self.required_indicators: list[Indicator] = []

        def on_bar(self, ctx: Context) -> list[Signal]:
            signal = self._script.signals_by_bar.get(ctx.bar_index)
            return [signal] if signal is not None else []

    _Fake.__name__ = f"FakeChild_{child_id}"
    return _Fake


def _fake_child(*, script: FakeScript, child_id: str = "fake_child") -> Strategy:
    """Construct a FakeChild instance with the requested id."""
    cls = _make_fake_child_class(child_id)
    return cls(script=script)  # type: ignore[call-arg]


def _ctx(bar_index: int) -> Context:
    """Build a minimal synthetic ``Context`` for compositestrategy tests."""
    ts = datetime(2024, 1, 1, 9, 15, tzinfo=IST) + pd.Timedelta(minutes=bar_index)
    return Context(
        symbol="SYNTH",
        bars=pd.DataFrame(
            {
                "timestamp": [ts],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
            }
        ),
        bar_index=0,
        timestamp=ts,
        timeframe="1m",
    )


def _signal(
    side: str,
    *,
    entry: float = 100.0,
    stop_loss: float | None = None,
    target: float | None = None,
    confidence: float = 0.5,
    strategy_id: str = "fake_child",
    reasons: list[str] | None = None,
    indicator_snapshot: dict[str, float] | None = None,
    ts: datetime | None = None,
) -> Signal:
    """Build a Signal with sensible defaults for the given side."""
    if side == "BUY":
        sl = stop_loss if stop_loss is not None else entry - 1.0
        tgt = target if target is not None else entry + 2.0
    else:
        sl = stop_loss if stop_loss is not None else entry + 1.0
        tgt = target if target is not None else entry - 2.0
    return Signal(
        symbol="SYNTH",
        side=side,  # type: ignore[arg-type]
        entry=entry,
        stop_loss=sl,
        target=tgt,
        timeframe="1m",
        strategy_id=strategy_id,
        reasons=reasons if reasons is not None else [f"{side} from {strategy_id}"],
        indicator_snapshot=indicator_snapshot or {},
        confidence=confidence,
        ts=ts or datetime(2024, 1, 1, 9, 15, tzinfo=IST),
    )


# ---------------------------------------------------------------------------
# direction policy logic
# ---------------------------------------------------------------------------


def test_unanimous_all_agree_emits_signal() -> None:
    """Three BUY votes from three children → composite emits BUY."""
    children = [
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c1")}), child_id="c1"),
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c2")}), child_id="c2"),
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c3")}), child_id="c3"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="unanimous")
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert signals[0].side == "BUY"


def test_unanimous_abstention_breaks_unanimity() -> None:
    """2 BUY + 1 abstain → no signal under unanimous."""
    children = [
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c1")}), child_id="c1"),
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c2")}), child_id="c2"),
        _fake_child(script=FakeScript({}), child_id="c3"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="unanimous")
    )
    assert composite.on_bar(_ctx(0)) == []


def test_unanimous_opposing_vote_vetoes() -> None:
    """2 BUY + 1 SELL → no signal under unanimous."""
    children = [
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c1")}), child_id="c1"),
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c2")}), child_id="c2"),
        _fake_child(script=FakeScript({0: _signal("SELL", strategy_id="c3")}), child_id="c3"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="unanimous")
    )
    assert composite.on_bar(_ctx(0)) == []


def test_majority_opposite_vote_vetoes() -> None:
    """2 BUY + 1 SELL → no signal under majority (opposing vetoes)."""
    children = [
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c1")}), child_id="c1"),
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c2")}), child_id="c2"),
        _fake_child(script=FakeScript({0: _signal("SELL", strategy_id="c3")}), child_id="c3"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="majority")
    )
    assert composite.on_bar(_ctx(0)) == []


def test_majority_two_buys_one_abstain_fires() -> None:
    """2 BUY + 1 abstain → BUY under majority."""
    children = [
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c1")}), child_id="c1"),
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c2")}), child_id="c2"),
        _fake_child(script=FakeScript({}), child_id="c3"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="majority")
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert signals[0].side == "BUY"


def test_majority_split_one_each_abstain_no_signal() -> None:
    """1 BUY + 1 SELL + 1 abstain → no signal under majority."""
    children = [
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c1")}), child_id="c1"),
        _fake_child(script=FakeScript({0: _signal("SELL", strategy_id="c2")}), child_id="c2"),
        _fake_child(script=FakeScript({}), child_id="c3"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="majority")
    )
    assert composite.on_bar(_ctx(0)) == []


def test_any_single_fire_no_opposite() -> None:
    """1 BUY + others abstain → BUY under any."""
    children = [
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c1")}), child_id="c1"),
        _fake_child(script=FakeScript({}), child_id="c2"),
        _fake_child(script=FakeScript({}), child_id="c3"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="any")
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert signals[0].side == "BUY"


def test_any_conflicting_votes_no_signal() -> None:
    """1 BUY + 1 SELL → no signal under any (conflict vetoes)."""
    children = [
        _fake_child(script=FakeScript({0: _signal("BUY", strategy_id="c1")}), child_id="c1"),
        _fake_child(script=FakeScript({0: _signal("SELL", strategy_id="c2")}), child_id="c2"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="any")
    )
    assert composite.on_bar(_ctx(0)) == []


# ---------------------------------------------------------------------------
# Empty bars (no votes at all) — composite must not crash, must abstain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["unanimous", "majority", "any"])
def test_no_children_vote_no_signal(direction: str) -> None:
    children = [
        _fake_child(script=FakeScript({}), child_id=f"c{i}") for i in range(3)
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction=direction)  # type: ignore[arg-type]
    )
    assert composite.on_bar(_ctx(0)) == []


# ---------------------------------------------------------------------------
# price aggregation
# ---------------------------------------------------------------------------


def test_long_tightest_picks_closest_levels() -> None:
    """Long tightest = max(SLs), min(targets)."""
    children = [
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", entry=100.0, stop_loss=95.0, target=110.0, strategy_id="c1",
            )}),
            child_id="c1",
        ),
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", entry=100.0, stop_loss=97.0, target=108.0, strategy_id="c2",
            )}),
            child_id="c2",
        ),
    ]
    composite = CompositeStrategy(
        children=children,
        policy=CombinePolicy(direction="any", price="tightest"),
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert signals[0].stop_loss == pytest.approx(97.0)
    assert signals[0].target == pytest.approx(108.0)


def test_long_widest_picks_furthest_levels() -> None:
    """Long widest = min(SLs), max(targets)."""
    children = [
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", entry=100.0, stop_loss=95.0, target=110.0, strategy_id="c1",
            )}),
            child_id="c1",
        ),
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", entry=100.0, stop_loss=97.0, target=108.0, strategy_id="c2",
            )}),
            child_id="c2",
        ),
    ]
    composite = CompositeStrategy(
        children=children,
        policy=CombinePolicy(direction="any", price="widest"),
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert signals[0].stop_loss == pytest.approx(95.0)
    assert signals[0].target == pytest.approx(110.0)


def test_long_average_means_levels() -> None:
    children = [
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", entry=100.0, stop_loss=95.0, target=110.0, strategy_id="c1",
            )}),
            child_id="c1",
        ),
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", entry=100.0, stop_loss=97.0, target=108.0, strategy_id="c2",
            )}),
            child_id="c2",
        ),
    ]
    composite = CompositeStrategy(
        children=children,
        policy=CombinePolicy(direction="any", price="average"),
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert signals[0].stop_loss == pytest.approx(96.0)
    assert signals[0].target == pytest.approx(109.0)


def test_short_tightest_flips_comparison() -> None:
    """Short tightest = min(SLs above entry), max(targets below entry)."""
    children = [
        _fake_child(
            script=FakeScript({0: _signal(
                "SELL", entry=100.0, stop_loss=105.0, target=90.0, strategy_id="c1",
            )}),
            child_id="c1",
        ),
        _fake_child(
            script=FakeScript({0: _signal(
                "SELL", entry=100.0, stop_loss=103.0, target=92.0, strategy_id="c2",
            )}),
            child_id="c2",
        ),
    ]
    composite = CompositeStrategy(
        children=children,
        policy=CombinePolicy(direction="any", price="tightest"),
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert signals[0].stop_loss == pytest.approx(103.0)
    assert signals[0].target == pytest.approx(92.0)


def test_short_widest_flips_comparison() -> None:
    children = [
        _fake_child(
            script=FakeScript({0: _signal(
                "SELL", entry=100.0, stop_loss=105.0, target=90.0, strategy_id="c1",
            )}),
            child_id="c1",
        ),
        _fake_child(
            script=FakeScript({0: _signal(
                "SELL", entry=100.0, stop_loss=103.0, target=92.0, strategy_id="c2",
            )}),
            child_id="c2",
        ),
    ]
    composite = CompositeStrategy(
        children=children,
        policy=CombinePolicy(direction="any", price="widest"),
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert signals[0].stop_loss == pytest.approx(105.0)
    assert signals[0].target == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# confidence + reason aggregation
# ---------------------------------------------------------------------------


def test_confidence_averages_only_winning_side() -> None:
    """Confidence ignores losing-side children entirely.

    With ``any`` policy, only winning children must vote (no opposite).
    Confidence is the mean of just the winners' confidences.
    """
    children = [
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", strategy_id="c1", confidence=0.4,
            )}),
            child_id="c1",
        ),
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", strategy_id="c2", confidence=0.8,
            )}),
            child_id="c2",
        ),
        _fake_child(script=FakeScript({}), child_id="c3"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="majority", price="tightest")
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert signals[0].confidence == pytest.approx(0.6)


def test_reasons_are_prefixed_with_child_tag() -> None:
    children = [
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", strategy_id="alpha", reasons=["fast over slow"],
            )}),
            child_id="alpha",
        ),
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", strategy_id="beta", reasons=["macd > signal"],
            )}),
            child_id="beta",
        ),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="unanimous")
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    assert "alpha: fast over slow" in signals[0].reasons
    assert "beta: macd > signal" in signals[0].reasons


def test_indicator_snapshot_namespaced_by_child_tag() -> None:
    children = [
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", strategy_id="alpha",
                indicator_snapshot={"rsi": 25.0, "atr": 1.5},
            )}),
            child_id="alpha",
        ),
        _fake_child(
            script=FakeScript({0: _signal(
                "BUY", strategy_id="beta",
                indicator_snapshot={"macd": 0.7},
            )}),
            child_id="beta",
        ),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="unanimous")
    )
    signals = composite.on_bar(_ctx(0))
    assert len(signals) == 1
    snap = signals[0].indicator_snapshot
    assert snap["alpha.rsi"] == 25.0
    assert snap["alpha.atr"] == 1.5
    assert snap["beta.macd"] == 0.7


# ---------------------------------------------------------------------------
# child disambiguation
# ---------------------------------------------------------------------------


def test_duplicate_child_ids_get_suffix() -> None:
    children = [
        _fake_child(script=FakeScript({}), child_id="ema_crossover"),
        _fake_child(script=FakeScript({}), child_id="ema_crossover"),
        _fake_child(script=FakeScript({}), child_id="rsi_mean_revert"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="unanimous")
    )
    assert composite.child_tags == (
        "ema_crossover_1",
        "ema_crossover_2",
        "rsi_mean_revert",
    )


def test_unique_child_ids_keep_raw_id() -> None:
    children = [
        _fake_child(script=FakeScript({}), child_id="alpha"),
        _fake_child(script=FakeScript({}), child_id="beta"),
    ]
    composite = CompositeStrategy(
        children=children, policy=CombinePolicy(direction="unanimous")
    )
    assert composite.child_tags == ("alpha", "beta")


# ---------------------------------------------------------------------------
# CompositeStrategy construction guard rails
# ---------------------------------------------------------------------------


def test_composite_rejects_lt_two_children() -> None:
    only_one = [_fake_child(script=FakeScript({}), child_id="solo")]
    with pytest.raises(ValueError, match="at least 2"):
        CompositeStrategy(children=only_one, policy=CombinePolicy())


# ---------------------------------------------------------------------------
# Aggregation helpers (direct calls for crisp regressions)
# ---------------------------------------------------------------------------


def test_resolve_direction_helper_unanimous_with_abstention() -> None:
    """Aggregate helper agrees with the strategy-level contract."""
    from dashboard.services.composite import _ChildVote  # internal type

    votes: list[_ChildVote] = [
        _ChildVote(tag="c1", signal=_signal("BUY", strategy_id="c1")),
        _ChildVote(tag="c2", signal=_signal("BUY", strategy_id="c2")),
    ]
    # Three children but only two voted → abstention vetoes unanimity.
    assert resolve_direction(votes, total_children=3, policy="unanimous") is None
    # Two children and both voted BUY → fires.
    assert resolve_direction(votes, total_children=2, policy="unanimous") == "BUY"


def test_aggregate_stop_target_average_is_arithmetic_mean() -> None:
    from dashboard.services.composite import _ChildVote

    votes = [
        _ChildVote(tag="c1", signal=_signal("BUY", stop_loss=90.0, target=120.0)),
        _ChildVote(tag="c2", signal=_signal("BUY", stop_loss=95.0, target=110.0)),
    ]
    assert aggregate_stop(winners=votes, side="BUY", policy="average") == pytest.approx(92.5)
    assert aggregate_target(winners=votes, side="BUY", policy="average") == pytest.approx(115.0)


# ---------------------------------------------------------------------------
# run_combined persistence (in-memory DuckDB)
# ---------------------------------------------------------------------------


def _dashboard_conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "dash.duckdb"))
    conn.execute(BACKTEST_RUNS_SCHEMA)
    conn.execute(BACKTEST_GROUPS_SCHEMA)
    conn.execute(STRATEGY_SETTINGS_SCHEMA)
    return conn


def test_run_combined_persists_single_row_with_composite_kind(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        children = [
            StrategySelection(
                strategy_id="ema_crossover",
                params={"fast_period": 5, "slow_period": 12, "atr_period": 7},
            ),
            StrategySelection(strategy_id="bbands_breakout", params={}),
        ]
        run_id = runner.run_combined(
            children=children,
            policy=CombinePolicy(direction="majority", price="average"),
            symbol="SYNTH",
            bars_count=200,
            seed=11,
        )
        assert isinstance(run_id, str) and len(run_id) == 12

        detail = runner.get_run(run_id)
        assert detail is not None
        # Composite uses the synthetic "composite" strategy id.
        assert detail.summary.strategy == "composite"
        # Combined runs are NOT part of a comparison group.
        assert detail.summary.group_id is None
        params = detail.summary.params
        assert params["kind"] == "composite"
        assert params["policy"] == {"direction": "majority", "price": "average"}
        stored_children = params["children"]
        assert {c["strategy"] for c in stored_children} == {
            "ema_crossover",
            "bbands_breakout",
        }
        # Source meta still flows through under the same key the
        # legacy run uses, so the detail page's source row works.
        assert params["source"]["type"] == "synthetic"
    finally:
        conn.close()


def test_run_combined_rejects_lt_two_children(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        with pytest.raises(ValueError, match="at least 2"):
            runner.run_combined(
                children=[
                    StrategySelection(strategy_id="ema_crossover", params={}),
                ],
                policy=CombinePolicy(),
                symbol="SYNTH",
                bars_count=120,
            )
    finally:
        conn.close()


def test_run_combined_rejects_unknown_strategy(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        with pytest.raises(KeyError):
            runner.run_combined(
                children=[
                    StrategySelection(strategy_id="nope", params={}),
                    StrategySelection(strategy_id="ema_crossover", params={}),
                ],
                policy=CombinePolicy(),
                symbol="SYNTH",
                bars_count=120,
            )
    finally:
        conn.close()


def test_run_combined_rejects_unknown_param(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        with pytest.raises(ValueError, match="unknown params"):
            runner.run_combined(
                children=[
                    StrategySelection(
                        strategy_id="ema_crossover", params={"not_a_param": 1}
                    ),
                    StrategySelection(strategy_id="bbands_breakout", params={}),
                ],
                policy=CombinePolicy(),
                symbol="SYNTH",
                bars_count=120,
            )
    finally:
        conn.close()


def test_run_combined_smoke_runs_two_children(tmp_path: Path) -> None:
    conn = _dashboard_conn(tmp_path)
    try:
        runner = BacktestRunner(conn)
        run_id = runner.run_combined(
            children=[
                StrategySelection(strategy_id="ema_crossover", params={}),
                StrategySelection(strategy_id="bbands_breakout", params={}),
            ],
            policy=CombinePolicy(direction="any", price="tightest"),
            symbol="SYNTH",
            bars_count=200,
            seed=7,
        )
        detail = runner.get_run(run_id)
        assert detail is not None
        assert detail.summary.bars_count == 200
        # Equity curve always recorded (one point per bar).
        assert len(detail.equity_curve) == 200
        # P&L is finite (no NaNs sneaking through aggregation).
        assert math.isfinite(detail.summary.total_pnl)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# /api/backtest/run-combined endpoint
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


def test_api_run_combined_returns_run_id(client: TestClient) -> None:
    payload = {
        "children": [
            {"strategy": "ema_crossover", "params": {"fast_period": 5}},
            {"strategy": "rsi_mean_revert", "params": {}},
            {"strategy": "bbands_breakout", "params": {}},
        ],
        "policy": {"direction": "majority", "price": "average"},
        "symbol": "SYNTH",
        "bars_count": 200,
        "seed": 17,
    }
    response = client.post("/api/backtest/run-combined", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "id" in body and isinstance(body["id"], str)

    # The persisted row is reachable from the detail endpoint.
    detail = client.get(f"/backtests/{body['id']}")
    assert detail.status_code == 200
    assert "Composite makeup" in detail.text


def test_api_run_combined_one_child_422(client: TestClient) -> None:
    response = client.post(
        "/api/backtest/run-combined",
        json={
            "children": [{"strategy": "ema_crossover"}],
            "symbol": "SYNTH",
            "bars_count": 100,
        },
    )
    assert response.status_code == 422


def test_api_run_combined_unknown_strategy_400(client: TestClient) -> None:
    response = client.post(
        "/api/backtest/run-combined",
        json={
            "children": [
                {"strategy": "made_up"},
                {"strategy": "ema_crossover"},
            ],
            "symbol": "SYNTH",
            "bars_count": 100,
        },
    )
    assert response.status_code == 400
    assert "made_up" in response.json()["detail"]


def test_api_run_combined_unknown_param_400(client: TestClient) -> None:
    response = client.post(
        "/api/backtest/run-combined",
        json={
            "children": [
                {"strategy": "ema_crossover", "params": {"not_a_param": 5}},
                {"strategy": "bbands_breakout"},
            ],
            "symbol": "SYNTH",
            "bars_count": 100,
        },
    )
    assert response.status_code == 400
    assert "not_a_param" in response.json()["detail"]


def test_api_run_combined_bad_policy_direction_422(client: TestClient) -> None:
    response = client.post(
        "/api/backtest/run-combined",
        json={
            "children": [
                {"strategy": "ema_crossover"},
                {"strategy": "bbands_breakout"},
            ],
            "policy": {"direction": "everybody", "price": "tightest"},
            "symbol": "SYNTH",
            "bars_count": 100,
        },
    )
    assert response.status_code == 422


def test_api_run_combined_duplicate_strategy_ids_are_allowed(
    client: TestClient,
) -> None:
    """Combine intentionally allows blending two parameterisations of one strategy."""
    response = client.post(
        "/api/backtest/run-combined",
        json={
            "children": [
                {"strategy": "ema_crossover", "params": {"fast_period": 5, "slow_period": 12}},
                {"strategy": "ema_crossover", "params": {"fast_period": 8, "slow_period": 21}},
            ],
            "policy": {"direction": "any", "price": "tightest"},
            "symbol": "SYNTH",
            "bars_count": 150,
        },
    )
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# detail page renders composite makeup card
# ---------------------------------------------------------------------------


def test_detail_page_renders_composite_makeup(
    client: TestClient, app_state: AppState
) -> None:
    runner = BacktestRunner(app_state.dashboard_conn(), settings=app_state.settings)
    run_id = runner.run_combined(
        children=[
            StrategySelection(strategy_id="ema_crossover", params={}),
            StrategySelection(strategy_id="ema_crossover", params={"fast_period": 5}),
            StrategySelection(strategy_id="bbands_breakout", params={}),
        ],
        policy=CombinePolicy(direction="majority", price="widest"),
        symbol="SYNTH",
        bars_count=180,
        seed=3,
    )
    page = client.get(f"/backtests/{run_id}")
    assert page.status_code == 200
    body = page.text
    assert "Composite makeup" in body
    assert "direction: majority" in body
    assert "price: widest" in body
    # Disambiguation suffixes appear for duplicate ids.
    assert "ema_crossover_1" in body
    assert "ema_crossover_2" in body


def test_backtests_page_includes_combine_mode_toggle(client: TestClient) -> None:
    response = client.get("/backtests")
    assert response.status_code == 200
    body = response.text
    # The new mode toggle and policy panel render.
    assert 'id="bt-mode-toggle"' in body
    assert 'data-value="combine"' in body
    assert 'id="bt-policy-panel"' in body
    # Direction radios for each policy option are present.
    assert 'value="unanimous"' in body
    assert 'value="majority"' in body
    assert 'value="any"' in body
    # And so are price radios.
    assert 'value="tightest"' in body
    assert 'value="widest"' in body
    assert 'value="average"' in body
    # Submit JS knows about the new endpoint.
    assert "/api/backtest/run-combined" in body
