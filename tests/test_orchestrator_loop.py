"""Tests for the OrchestratorLoop symbol routing and qty cap."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pandas as pd
import pytest

from config.settings import AppSettings
from core.bar import Bar
from core.context import Context
from core.signal import Signal
from data.feed import BarFeed
from execution.paper import PaperBroker
from orchestrator.loop import OrchestratorLoop
from risk.manager import RiskManager


class _StubStrategy:
    """Strategy that emits one signal on bar_index=0 then stays silent."""

    id = "stub_strategy"
    timeframe = "1m"

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self._emitted = False

    def on_bar(self, ctx: Context) -> list[Signal]:
        if self._emitted:
            return []
        self._emitted = True
        row = ctx.bars.iloc[ctx.bar_index]
        return [
            Signal(
                symbol=self._symbol,
                side="BUY",
                entry=float(row["close"]),
                stop_loss=float(row["close"]) - 1.0,
                target=float(row["close"]) + 1.0,
                timeframe=self.timeframe,
                strategy_id=self.id,
                reasons=["test"],
                indicator_snapshot={"ema": 1.0},
                confidence=0.7,
                ts=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            )
        ]


def _bar_frame(bar_symbol: str = "SYNTH") -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2024-01-01 09:15:00",
        periods=3,
        freq="1min",
        tz="Asia/Kolkata",
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.0, 100.0],
            "high": [100.5, 102.0, 100.5],
            "low": [99.5, 100.0, 99.5],
            "close": [100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 1000.0],
            "symbol": [bar_symbol, bar_symbol, bar_symbol],
        }
    )


class _FrameFeed(BarFeed):
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def bars(self) -> Iterator[Bar]:
        for _, row in self._frame.iterrows():
            yield Bar(
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy()


def test_loop_exits_using_signal_symbol_not_bar_symbol() -> None:
    """Regression: bar frame symbol differs from strategy symbol.

    The strategy emits a BUY for RELIANCE while bars carry symbol=SYNTH.
    The next bar's high (102.0) crosses the target (101.0); the broker
    holds the position keyed by RELIANCE. The loop must look up by
    signal.symbol — not the bar frame symbol — to exit correctly.
    """
    frame = _bar_frame(bar_symbol="SYNTH")
    strategy = _StubStrategy(symbol="RELIANCE")
    broker = PaperBroker(settings=AppSettings())
    loop = OrchestratorLoop(
        strategy=strategy,
        broker=broker,
        risk=RiskManager(AppSettings()),
        feed=_FrameFeed(frame),
    )
    result = asyncio.run(loop.run())
    assert result.stats.orders_placed == 1
    assert result.stats.bar_exits == 1
    assert result.open_positions == 0
    # Position was stored under the strategy symbol.
    assert broker.get_positions() == []


def test_loop_override_qty_caps_order_size() -> None:
    frame = _bar_frame()
    strategy = _StubStrategy(symbol="SYNTH")
    broker = PaperBroker(settings=AppSettings())
    loop = OrchestratorLoop(
        strategy=strategy,
        broker=broker,
        risk=RiskManager(AppSettings()),
        feed=_FrameFeed(frame),
        override_qty=1,
    )
    result = asyncio.run(loop.run())
    assert result.stats.orders_placed == 1
    placed = broker.orders[0]
    assert placed.qty == 1


def test_loop_override_qty_zero_rejects_signal() -> None:
    frame = _bar_frame()
    strategy = _StubStrategy(symbol="SYNTH")
    broker = PaperBroker(settings=AppSettings())
    loop = OrchestratorLoop(
        strategy=strategy,
        broker=broker,
        risk=RiskManager(AppSettings()),
        feed=_FrameFeed(frame),
        override_qty=0,
    )
    result = asyncio.run(loop.run())
    assert result.stats.orders_placed == 0
    assert result.stats.risk_rejected >= 1


def test_strategy_symbol_used_in_context() -> None:
    """The context symbol fed to the strategy is the user-requested symbol."""
    observed: list[str] = []

    class _Recorder(_StubStrategy):
        def on_bar(self, ctx: Context) -> list[Signal]:
            observed.append(ctx.symbol)
            return []

    frame = _bar_frame(bar_symbol="SYNTH")
    strategy = _Recorder(symbol="RELIANCE")
    broker = PaperBroker(settings=AppSettings())
    loop = OrchestratorLoop(
        strategy=strategy,
        broker=broker,
        risk=RiskManager(AppSettings()),
        feed=_FrameFeed(frame),
    )
    asyncio.run(loop.run())
    assert observed
    assert all(s == "RELIANCE" for s in observed)


@pytest.mark.parametrize("invalid", [-3, -1])
def test_loop_negative_override_qty_rejects(invalid: int) -> None:
    frame = _bar_frame()
    strategy = _StubStrategy(symbol="SYNTH")
    broker = PaperBroker(settings=AppSettings())
    loop = OrchestratorLoop(
        strategy=strategy,
        broker=broker,
        risk=RiskManager(AppSettings()),
        feed=_FrameFeed(frame),
        override_qty=invalid,
    )
    result = asyncio.run(loop.run())
    assert result.stats.orders_placed == 0
