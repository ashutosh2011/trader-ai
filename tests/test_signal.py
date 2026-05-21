"""Direction validation and tick-rounding tests for :class:`Signal`."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import strategies  # noqa: F401 — register example strategies
from core.context import Context
from core.signal import Signal
from strategies.examples.ema_crossover import EmaCrossover
from tests.fixtures.bars import make_synthetic_bars

IST = ZoneInfo("Asia/Kolkata")


def _make_signal(**overrides: object) -> Signal:
    defaults: dict[str, object] = {
        "symbol": "SYNTH",
        "side": "BUY",
        "entry": 100.0,
        "stop_loss": 99.0,
        "target": 103.0,
        "timeframe": "1m",
        "strategy_id": "test",
        "reasons": ["test"],
        "indicator_snapshot": {"ema": 1.0},
        "confidence": 0.7,
        "ts": datetime(2024, 1, 1, 10, 0, tzinfo=IST),
    }
    defaults.update(overrides)
    return Signal(**defaults)  # type: ignore[arg-type]


def test_buy_signal_accepted() -> None:
    signal = _make_signal(side="BUY", entry=100.0, stop_loss=99.0, target=103.0)
    assert signal.side == "BUY"
    assert signal.stop_loss < signal.entry < signal.target


def test_sell_signal_accepted() -> None:
    signal = _make_signal(side="SELL", entry=100.0, stop_loss=101.0, target=97.0)
    assert signal.side == "SELL"
    assert signal.target < signal.entry < signal.stop_loss


def test_buy_with_inverted_stop_raises() -> None:
    with pytest.raises(ValueError, match="BUY signal requires"):
        _make_signal(side="BUY", entry=100.0, stop_loss=101.0, target=105.0)


def test_buy_with_target_below_entry_raises() -> None:
    with pytest.raises(ValueError, match="BUY signal requires"):
        _make_signal(side="BUY", entry=100.0, stop_loss=99.0, target=99.5)


def test_sell_with_inverted_stop_raises() -> None:
    with pytest.raises(ValueError, match="SELL signal requires"):
        _make_signal(side="SELL", entry=100.0, stop_loss=99.0, target=97.0)


def test_negative_entry_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _make_signal(side="BUY", entry=-1.0, stop_loss=-2.0, target=5.0)


def test_zero_stop_loss_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _make_signal(side="BUY", entry=100.0, stop_loss=0.0, target=103.0)


def test_existing_strategy_signals_still_pass() -> None:
    """Smoke check: live EMA-crossover signals continue to validate."""
    frame = make_synthetic_bars(200, seed=42)
    strategy = EmaCrossover(fast_period=5, slow_period=10, atr_period=5, symbol="SYNTH")
    signals_seen = 0
    for bar_index in range(strategy.warmup_bars() + 1, len(frame)):
        ctx = Context(
            symbol="SYNTH",
            bars=frame,
            bar_index=bar_index,
            timestamp=frame["timestamp"].iloc[bar_index].to_pydatetime(),
            timeframe="1m",
        )
        for signal in strategy.on_bar(ctx):
            signals_seen += 1
            if signal.side == "BUY":
                assert signal.stop_loss < signal.entry < signal.target
            else:
                assert signal.target < signal.entry < signal.stop_loss
    assert signals_seen >= 1


def test_signal_with_tick_rounding_default_005() -> None:
    """Basic round of a BUY signal onto the 0.05 tick grid."""
    signal = _make_signal(side="BUY", entry=100.12, stop_loss=99.07, target=102.83)
    rounded = signal.with_tick_rounding(0.05)
    assert rounded.entry == pytest.approx(100.10, abs=1e-9)
    # BUY stop floors → 99.07 → 99.05.
    assert rounded.stop_loss == pytest.approx(99.05, abs=1e-9)
    # BUY target floors → 102.83 → 102.80.
    assert rounded.target == pytest.approx(102.80, abs=1e-9)
    # Original signal is left untouched.
    assert signal.entry == pytest.approx(100.12, abs=1e-9)


def test_signal_with_tick_rounding_entry_nearest() -> None:
    """tick=0.05 with entry=100.07 → 100.05 (nearest)."""
    signal = _make_signal(side="BUY", entry=100.07, stop_loss=99.50, target=103.50)
    rounded = signal.with_tick_rounding(0.05)
    assert rounded.entry == pytest.approx(100.05, abs=1e-9)


def test_signal_with_tick_rounding_buy_stop_floors() -> None:
    """BUY stop 99.93 with tick 0.05 → 99.90 (floor)."""
    signal = _make_signal(side="BUY", entry=100.50, stop_loss=99.93, target=103.00)
    rounded = signal.with_tick_rounding(0.05)
    assert rounded.stop_loss == pytest.approx(99.90, abs=1e-9)


def test_signal_with_tick_rounding_sell_stop_ceils_target_ceils() -> None:
    """SELL stop ceils up (wider), SELL target ceils up (closer)."""
    signal = _make_signal(side="SELL", entry=100.07, stop_loss=101.02, target=98.13)
    rounded = signal.with_tick_rounding(0.05)
    assert rounded.entry == pytest.approx(100.05, abs=1e-9)
    assert rounded.stop_loss == pytest.approx(101.05, abs=1e-9)
    assert rounded.target == pytest.approx(98.15, abs=1e-9)


def test_signal_with_tick_rounding_violation_raises() -> None:
    """Tick larger than the BUY spread collapses the order and must raise."""
    signal = _make_signal(side="BUY", entry=100.04, stop_loss=100.01, target=100.06)
    with pytest.raises(ValueError, match="BUY signal requires"):
        signal.with_tick_rounding(0.50)


def test_signal_with_tick_rounding_invalid_tick_raises() -> None:
    signal = _make_signal()
    with pytest.raises(ValueError, match="tick_size must be positive"):
        signal.with_tick_rounding(0.0)


def test_signal_with_tick_rounding_preserves_metadata() -> None:
    signal = _make_signal(
        side="BUY",
        entry=100.07,
        stop_loss=99.93,
        target=102.83,
        qty=10,
    )
    rounded = signal.with_tick_rounding(0.05)
    assert rounded.symbol == signal.symbol
    assert rounded.timeframe == signal.timeframe
    assert rounded.strategy_id == signal.strategy_id
    assert rounded.reasons == signal.reasons
    assert rounded.indicator_snapshot == signal.indicator_snapshot
    assert rounded.confidence == pytest.approx(signal.confidence)
    assert rounded.ts == signal.ts
    assert rounded.qty == signal.qty
