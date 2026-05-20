from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config.settings import AppSettings, RiskConfig
from core.context import Context
from core.signal import Signal
from risk.manager import RiskManager, RiskState, is_in_no_trade_window, is_kill_switch_active
from tests.fixtures.bars import make_synthetic_bars

IST = ZoneInfo("Asia/Kolkata")


def _signal(ts: datetime, *, entry: float = 100.0, stop: float = 99.0) -> Signal:
    return Signal(
        symbol="SYNTH",
        side="BUY",
        entry=entry,
        stop_loss=stop,
        target=103.0,
        timeframe="1m",
        strategy_id="test",
        reasons=["test"],
        indicator_snapshot={"ema": 1.0},
        confidence=0.8,
        ts=ts,
    )


def _ctx(frame: pd.DataFrame) -> Context:
    return Context(
        symbol="SYNTH",
        bars=frame,
        bar_index=len(frame) - 1,
        timestamp=frame["timestamp"].iloc[-1].to_pydatetime(),
        timeframe="1m",
    )


def test_kill_switch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KILL_SWITCH", "1")
    assert is_kill_switch_active(kill_file=Path("/nonexistent/KILL"))


def test_kill_switch_file(tmp_path: Path) -> None:
    kill = tmp_path / "KILL"
    kill.touch()
    assert is_kill_switch_active(kill_file=kill)


def test_no_trade_window() -> None:
    ts = datetime(2024, 1, 1, 9, 20, tzinfo=IST)
    assert is_in_no_trade_window(ts, ["09:15-09:30"])
    ts_ok = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    assert not is_in_no_trade_window(ts_ok, ["09:15-09:30"])


def test_daily_loss_cap_rejects() -> None:
    settings = AppSettings(risk=RiskConfig(daily_loss_cap_pct=1.0))
    rm = RiskManager(settings)
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0, daily_realized_pnl=-1500.0)
    decision = rm.pre_check(_signal(ts), ctx, state)
    assert not decision.approved
    assert decision.reason == "daily_loss_cap"


def test_max_open_positions() -> None:
    settings = AppSettings(risk=RiskConfig(max_open_positions=1))
    rm = RiskManager(settings)
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0, open_positions=1)
    decision = rm.pre_check(_signal(ts), ctx, state)
    assert not decision.approved
    assert decision.reason == "max_open_positions"


def test_atr_based_sizing() -> None:
    rm = RiskManager(AppSettings(risk=RiskConfig(max_loss_per_trade_pct=1.0)))
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0)
    signal = _signal(ts, entry=100.0, stop=99.0)
    qty = rm.size(signal, ctx, state, size_multiplier=1.0)
    assert qty == 1000
    qty_half = rm.size(signal, ctx, state, size_multiplier=0.5)
    assert qty_half == 500


def test_fixed_pct_sizing() -> None:
    settings = AppSettings(
        risk=RiskConfig(position_sizing="fixed_pct", fixed_position_pct=10.0)
    )
    rm = RiskManager(settings)
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0)
    qty = rm.size(_signal(ts), ctx, state)
    assert qty == 100
