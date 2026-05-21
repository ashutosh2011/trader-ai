from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config.settings import AppSettings, RiskConfig
from core.context import Context
from core.instrument import Instrument
from core.signal import Signal
from risk.manager import RiskManager, RiskState, is_in_no_trade_window, is_kill_switch_active
from tests.fixtures.bars import make_synthetic_bars

IST = ZoneInfo("Asia/Kolkata")


def _signal(
    ts: datetime,
    *,
    entry: float = 100.0,
    stop: float = 99.0,
    target: float = 103.0,
) -> Signal:
    return Signal(
        symbol="SYNTH",
        side="BUY",
        entry=entry,
        stop_loss=stop,
        target=target,
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


def test_kill_switch_file_absolute_path_survives_cwd_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kill = tmp_path / "KILL"
    kill.touch()
    monkeypatch.setenv("KILL_SWITCH_FILE", str(kill))

    settings = AppSettings()
    assert settings.kill_switch_file == kill
    assert settings.kill_switch_file.is_absolute()

    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    assert is_kill_switch_active(kill_file=settings.kill_switch_file)


def test_kill_switch_file_default_is_absolute() -> None:
    settings = AppSettings()
    assert settings.kill_switch_file.is_absolute()
    assert settings.kill_switch_file.name == "KILL"


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
    assert decision.reason == "daily_loss_cap_mtm"


def test_daily_pnl_mtm_property() -> None:
    state = RiskState(
        account_equity=100_000.0,
        daily_realized_pnl=-300.0,
        daily_unrealized_pnl=-200.0,
    )
    assert state.daily_pnl_mtm == pytest.approx(-500.0)

    state.daily_realized_pnl = 0.0
    state.daily_unrealized_pnl = 250.0
    assert state.daily_pnl_mtm == pytest.approx(250.0)


def test_daily_loss_cap_rejects_on_unrealized_only() -> None:
    settings = AppSettings(risk=RiskConfig(daily_loss_cap_pct=1.0))
    rm = RiskManager(settings)
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(
        account_equity=100_000.0,
        daily_realized_pnl=0.0,
        daily_unrealized_pnl=-1500.0,
    )
    decision = rm.pre_check(_signal(ts), ctx, state)
    assert not decision.approved
    assert decision.reason == "daily_loss_cap_mtm"


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


def test_size_floors_to_instrument_lot_size() -> None:
    """Nifty futures lot_size=75 → base_qty 200 floors to 150 (two lots)."""
    settings = AppSettings(risk=RiskConfig(max_loss_per_trade_pct=0.2))
    rm = RiskManager(settings)
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0)
    nifty_fut = Instrument.nse_future(
        "NIFTY24MAYFUT",
        date(2024, 5, 30),
        underlying="NIFTY",
        lot_size=75,
    )
    signal = _signal(ts, entry=100.0, stop=99.0)
    base = rm.size(signal, ctx, state)
    assert base == 200
    qty = rm.size(signal, ctx, state, instrument=nifty_fut)
    assert qty == 150


def test_size_zero_multiplier_returns_zero() -> None:
    """size_multiplier=0.0 must yield qty=0 (no silent floor to one)."""
    rm = RiskManager(AppSettings(risk=RiskConfig(max_loss_per_trade_pct=1.0)))
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0)
    qty = rm.size(_signal(ts), ctx, state, size_multiplier=0.0)
    assert qty == 0


def test_post_check_rejects_qty_zero_below_lot_size() -> None:
    """When the risk budget cannot afford a single lot, reject with that reason."""
    rm = RiskManager(AppSettings(risk=RiskConfig(max_loss_per_trade_pct=0.05)))
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0)
    instrument = Instrument.nse_future(
        "NIFTY24MAYFUT",
        date(2024, 5, 30),
        underlying="NIFTY",
        lot_size=75,
    )
    signal = _signal(ts, entry=100.0, stop=99.0)
    qty = rm.size(signal, ctx, state, instrument=instrument)
    assert qty == 0
    decision = rm.post_check(signal, ctx, state, qty, instrument=instrument)
    assert not decision.approved
    assert decision.reason == "qty_zero_below_lot_size"


def test_post_check_rejects_qty_zero_from_shrink() -> None:
    """When the analyst multiplier shrinks an otherwise-fine size below one lot."""
    rm = RiskManager(AppSettings(risk=RiskConfig(max_loss_per_trade_pct=1.0)))
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0)
    signal = _signal(ts, entry=100.0, stop=99.0)
    qty = rm.size(signal, ctx, state, size_multiplier=0.0)
    assert qty == 0
    decision = rm.post_check(signal, ctx, state, qty)
    assert not decision.approved
    assert decision.reason == "qty_zero_from_shrink"


def test_post_check_notional_cap_skipped_for_equity() -> None:
    """Equity instrument (or no instrument) bypasses the option premium cap."""
    rm = RiskManager(
        AppSettings(
            risk=RiskConfig(
                max_loss_per_trade_pct=2.0,
                max_premium_per_trade_pct=0.01,  # absurdly tight
            )
        )
    )
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0)
    equity = Instrument.nse_equity("RELIANCE")
    signal = _signal(ts, entry=100.0, stop=99.0, target=102.0)
    decision = rm.post_check(signal, ctx, state, qty=1, instrument=equity)
    assert decision.approved


def test_post_check_notional_cap_allows_option_under_cap() -> None:
    rm = RiskManager(
        AppSettings(
            risk=RiskConfig(
                max_loss_per_trade_pct=2.0,
                max_premium_per_trade_pct=1.0,
            )
        )
    )
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=1_000_000.0)
    option = Instrument.nse_option(
        "NIFTY24MAY24000CE",
        date(2024, 5, 30),
        strike=24000.0,
        option_type="CE",
        underlying="NIFTY",
        lot_size=50,
    )
    # premium = 100 * 1 * 50 = 5000, cap = 10_000 → approved
    signal = _signal(ts, entry=100.0, stop=80.0, target=140.0)
    decision = rm.post_check(signal, ctx, state, qty=1, instrument=option)
    assert decision.approved


def test_post_check_notional_cap_rejects_over_cap() -> None:
    rm = RiskManager(
        AppSettings(
            risk=RiskConfig(
                max_loss_per_trade_pct=2.0,
                max_premium_per_trade_pct=0.5,
            )
        )
    )
    frame = make_synthetic_bars(10)
    ctx = _ctx(frame)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    state = RiskState(account_equity=100_000.0)
    option = Instrument.nse_option(
        "NIFTY24MAY24000CE",
        date(2024, 5, 30),
        strike=24000.0,
        option_type="CE",
        underlying="NIFTY",
        lot_size=50,
    )
    # premium = 100 * 1 * 50 = 5000, cap = 500 → rejected
    signal = _signal(ts, entry=100.0, stop=80.0, target=140.0)
    decision = rm.post_check(signal, ctx, state, qty=1, instrument=option)
    assert not decision.approved
    assert decision.reason == "notional_cap_exceeded"
