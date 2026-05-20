from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtest.engine import (
    BacktestEngine,
    _check_exit,
    _fill_pending,
    _OpenPosition,
    _PendingEntry,
)
from core.context import Context
from core.signal import Signal
from indicators.base import Indicator
from strategies.base import Strategy
from strategies.examples.ema_crossover import EmaCrossover
from tests.fixtures.bars import make_synthetic_bars

IST = ZoneInfo("Asia/Kolkata")


def test_next_bar_open_fill() -> None:
    frame = _three_bar_frame()
    engine = BacktestEngine(qty=1)
    strategy = _SingleSignalStrategy(entry_close=100.0, target=103.0)
    result = engine.run(strategy, frame)
    assert result.trade_count == 1
    trade = result.closed_trades[0]
    assert trade.entry_price == 101.0
    assert trade.entry_bar == 1
    assert trade.exit_reason == "target"


def test_stop_loss_hit_long() -> None:
    position = _OpenPosition(
        symbol="X",
        side="LONG",
        entry_price=100.0,
        stop_loss=98.0,
        target=110.0,
        qty=1,
        entry_bar=0,
    )
    row = pd.Series({"high": 101.0, "low": 97.5})
    closed_position, trade = _check_exit(position=position, row=row, bar_index=2)
    assert closed_position is None
    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == 98.0
    assert trade.pnl == pytest.approx(-2.0)


def test_target_hit_long() -> None:
    position = _OpenPosition(
        symbol="X",
        side="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        target=105.0,
        qty=2,
        entry_bar=0,
    )
    row = pd.Series({"high": 106.0, "low": 99.0})
    _, trade = _check_exit(position=position, row=row, bar_index=1)
    assert trade is not None
    assert trade.exit_reason == "target"
    assert trade.pnl == pytest.approx(10.0)


def test_same_bar_stop_before_target_conservative() -> None:
    position = _OpenPosition(
        symbol="X",
        side="LONG",
        entry_price=100.0,
        stop_loss=99.0,
        target=101.0,
        qty=1,
        entry_bar=0,
    )
    row = pd.Series({"high": 102.0, "low": 98.0})
    _, trade = _check_exit(position=position, row=row, bar_index=1)
    assert trade is not None
    assert trade.exit_reason == "stop_loss"


def test_signal_reverse_closes_then_opens() -> None:
    long_signal = _make_signal(side="BUY")
    position = _OpenPosition(
        symbol="X",
        side="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        target=110.0,
        qty=1,
        entry_bar=0,
    )
    pending = _PendingEntry(signal=long_signal, signal_bar=0)
    new_position, remaining, reverse = _fill_pending(
        pending=pending,
        position=position,
        open_price=102.0,
        bar_index=3,
        default_qty=1,
    )
    assert reverse is not None
    assert reverse.exit_reason == "signal_reverse"
    assert reverse.exit_price == 102.0
    assert new_position is not None
    assert new_position.side == "LONG"
    assert remaining is None


def test_equity_curve_tracks_realized_pnl() -> None:
    frame = make_synthetic_bars(120, seed=7)
    result = BacktestEngine(qty=1).run(EmaCrossover(symbol="SYNTH"), frame)
    assert len(result.equity_curve) == len(frame)
    assert result.equity_curve[-1].equity == pytest.approx(result.total_pnl)


def test_strategy_exception_does_not_crash(synthetic_bars_200: pd.DataFrame) -> None:
    engine = BacktestEngine()
    result = engine.run(_BrokenStrategy(), synthetic_bars_200)
    assert result.trade_count == 0
    assert len(result.equity_curve) == len(synthetic_bars_200)


def test_no_lookahead_closed_bars() -> None:
    frame = make_synthetic_bars(50, seed=1)
    seen_lengths: list[int] = []

    class _LengthProbeStrategy(EmaCrossover):
        def on_bar(self, ctx: Context) -> list[Signal]:
            seen_lengths.append(len(ctx.closed_bars))
            return super().on_bar(ctx)

    BacktestEngine().run(_LengthProbeStrategy(symbol="SYNTH"), frame)
    assert seen_lengths == [idx + 1 for idx in range(len(frame))]


def _three_bar_frame() -> pd.DataFrame:
    ts = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    return pd.DataFrame(
        {
            "timestamp": [ts, ts.replace(minute=16), ts.replace(minute=17)],
            "open": [100.0, 101.0, 101.0],
            "high": [100.5, 101.5, 105.0],
            "low": [99.5, 100.5, 100.5],
            "close": [100.0, 101.0, 104.0],
            "volume": [1000.0, 1000.0, 1000.0],
            "symbol": ["X", "X", "X"],
        }
    )



def _make_signal(*, side: str) -> Signal:
    ts = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    stop = 95.0 if side == "BUY" else 105.0
    target = 110.0 if side == "BUY" else 90.0
    return Signal(
        symbol="X",
        side=side,  # type: ignore[arg-type]
        entry=100.0,
        stop_loss=stop,
        target=target,
        timeframe="1m",
        strategy_id="test",
        reasons=["test"],
        indicator_snapshot={"x": 1.0},
        confidence=0.5,
        ts=ts,
    )


class _SingleSignalStrategy(Strategy):
    id = "single_signal"
    timeframe = "1m"
    required_indicators: list[Indicator] = []

    def __init__(self, entry_close: float, target: float) -> None:
        super().__init__()
        self._entry_close = entry_close
        self._target = target
        self._fired = False

    def on_bar(self, ctx: Context) -> list[Signal]:
        if self._fired or ctx.bar_index != 0:
            return []
        self._fired = True
        return [
            Signal(
                symbol="X",
                side="BUY",
                entry=self._entry_close,
                stop_loss=self._entry_close - 1.0,
                target=self._target,
                timeframe=self.timeframe,
                strategy_id=self.id,
                reasons=["test entry"],
                indicator_snapshot={},
                confidence=0.5,
                ts=ctx.timestamp,
            )
        ]


class _BrokenStrategy(Strategy):
    id = "broken"
    timeframe = "1m"
    required_indicators: list[Indicator] = []

    def on_bar(self, ctx: Context) -> list[Signal]:
        raise RuntimeError("boom")
