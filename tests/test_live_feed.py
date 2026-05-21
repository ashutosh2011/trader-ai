"""Tests for :class:`LiveKiteFeed` aggregation and replay fallback."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from data.live_feed import KITE_MODE_FULL, LiveKiteFeed, TickerProtocol
from tests.fixtures.bars import make_synthetic_bars

IST = ZoneInfo("Asia/Kolkata")


class MockKiteTicker:
    """In-memory KiteTicker stand-in used for live-feed tests.

    Records subscribe/set_mode calls and exposes ``push_ticks`` to drive
    the registered ``on_ticks`` callback synchronously.
    """

    def __init__(self) -> None:
        self.subscribed: list[int] = []
        self.mode: str | None = None
        self.connect_calls = 0
        self.close_calls = 0
        self._on_ticks: Any = None
        self._on_connect: Any = None
        self._on_close: Any = None
        self._on_error: Any = None
        self._on_reconnect: Any = None

    def connect(self, threaded: bool = False) -> None:
        del threaded
        self.connect_calls += 1
        if self._on_connect is not None:
            self._on_connect(self, "ok")

    def close(self, code: int | None = None, reason: str | None = None) -> None:
        del code, reason
        self.close_calls += 1
        if self._on_close is not None:
            self._on_close(self, 1000, "shutdown")

    def subscribe(self, tokens: list[int]) -> None:
        self.subscribed.extend(tokens)

    def set_mode(self, mode: Any, tokens: list[int]) -> None:
        del tokens
        self.mode = str(mode)

    def on_ticks(self, callback: Any) -> None:
        self._on_ticks = callback

    def on_connect(self, callback: Any) -> None:
        self._on_connect = callback

    def on_close(self, callback: Any) -> None:
        self._on_close = callback

    def on_error(self, callback: Any) -> None:
        self._on_error = callback

    def on_reconnect(self, callback: Any) -> None:
        self._on_reconnect = callback

    def push_ticks(self, ticks: list[dict[str, Any]]) -> None:
        assert self._on_ticks is not None, "feed must be connected before pushing ticks"
        self._on_ticks(self, ticks)

    def trigger_reconnect(self, attempts: int = 1) -> None:
        assert self._on_reconnect is not None
        self._on_reconnect(self, attempts)

    def trigger_error(self, code: int = 4000, reason: str = "test") -> None:
        assert self._on_error is not None
        self._on_error(self, code, reason)

    def trigger_close(self, code: int = 1000, reason: str = "test") -> None:
        assert self._on_close is not None
        self._on_close(self, code, reason)


def _tick(
    token: int,
    ts: datetime,
    price: float,
    *,
    qty: int = 1,
) -> dict[str, Any]:
    return {
        "instrument_token": token,
        "last_price": price,
        "timestamp": ts,
        "last_traded_quantity": qty,
    }


@pytest.mark.asyncio
async def test_replay_mode_round_trip() -> None:
    bars = make_synthetic_bars(10, seed=1)
    feed = LiveKiteFeed(replay_source=bars)
    await feed.connect()
    assert feed.is_connected()
    df = feed.to_dataframe()
    assert len(df) == 10
    await feed.disconnect()
    assert not feed.is_connected()


@pytest.mark.asyncio
async def test_subscribes_full_mode_on_connect() -> None:
    ticker = MockKiteTicker()
    feed = LiveKiteFeed(
        instrument_tokens=[111, 222],
        symbol_map={111: "AAA", 222: "BBB"},
        ticker=ticker,
    )
    await feed.connect()
    assert ticker.connect_calls == 1
    assert ticker.subscribed == [111, 222]
    assert ticker.mode == KITE_MODE_FULL
    await feed.disconnect()
    assert ticker.close_calls == 1


@pytest.mark.asyncio
async def test_minute_boundary_emits_completed_bar() -> None:
    ticker = MockKiteTicker()
    feed = LiveKiteFeed(
        instrument_tokens=[1],
        symbol_map={1: "RELIANCE"},
        ticker=ticker,
    )
    await feed.connect()
    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=IST)
    # Three ticks in the 10:00 bucket, then one in 10:01.
    ticker.push_ticks(
        [
            _tick(1, base, 100.0),
            _tick(1, base + timedelta(seconds=15), 101.0),
            _tick(1, base + timedelta(seconds=45), 99.5),
        ]
    )
    # No bar closed yet — partial bucket still open.
    assert feed.to_dataframe().empty
    ticker.push_ticks([_tick(1, base + timedelta(minutes=1, seconds=2), 102.0)])
    df = feed.to_dataframe()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "RELIANCE"
    assert row["open"] == 100.0
    assert row["high"] == 101.0
    assert row["low"] == 99.5
    assert row["close"] == 99.5
    await feed.disconnect()


@pytest.mark.asyncio
async def test_multiple_symbols_emit_independently() -> None:
    ticker = MockKiteTicker()
    feed = LiveKiteFeed(
        instrument_tokens=[1, 2],
        symbol_map={1: "AAA", 2: "BBB"},
        ticker=ticker,
    )
    await feed.connect()
    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=IST)
    ticker.push_ticks(
        [
            _tick(1, base, 100.0),
            _tick(2, base, 200.0),
            _tick(1, base + timedelta(seconds=10), 101.0),
            _tick(2, base + timedelta(seconds=10), 201.0),
        ]
    )
    # Cross both boundaries.
    ticker.push_ticks(
        [
            _tick(1, base + timedelta(minutes=1), 102.0),
            _tick(2, base + timedelta(minutes=1), 199.0),
        ]
    )
    df = feed.to_dataframe()
    assert len(df) == 2
    by_symbol = {row["symbol"]: row for _, row in df.iterrows()}
    assert by_symbol["AAA"]["open"] == 100.0
    assert by_symbol["BBB"]["open"] == 200.0
    await feed.disconnect()


@pytest.mark.asyncio
async def test_reconnect_callback_does_not_crash() -> None:
    ticker = MockKiteTicker()
    feed = LiveKiteFeed(
        instrument_tokens=[1],
        symbol_map={1: "AAA"},
        ticker=ticker,
    )
    await feed.connect()
    ticker.trigger_reconnect(attempts=2)
    ticker.trigger_error(code=4001, reason="transient")
    ticker.trigger_close()
    # Feed should still accept new ticks after callbacks fire.
    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=IST)
    ticker.push_ticks([_tick(1, base, 100.0)])
    ticker.push_ticks([_tick(1, base + timedelta(minutes=1), 101.0)])
    df = feed.to_dataframe()
    assert len(df) == 1


@pytest.mark.asyncio
async def test_stream_bars_yields_via_queue() -> None:
    ticker = MockKiteTicker()
    feed = LiveKiteFeed(
        instrument_tokens=[1],
        symbol_map={1: "AAA"},
        ticker=ticker,
    )
    await feed.connect()
    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=IST)
    ticker.push_ticks([_tick(1, base, 100.0)])

    async def producer() -> None:
        await asyncio.sleep(0)
        ticker.push_ticks([_tick(1, base + timedelta(minutes=1), 101.0)])

    asyncio.create_task(producer())

    async def first() -> tuple[str, Any]:
        async for item in feed.stream_bars():
            return item
        msg = "stream_bars exited without yielding"
        raise AssertionError(msg)

    item = await asyncio.wait_for(first(), timeout=1.0)
    symbol, bar = item
    assert symbol == "AAA"
    assert bar.open == 100.0


@pytest.mark.asyncio
async def test_unsubscribed_token_is_ignored() -> None:
    ticker = MockKiteTicker()
    feed = LiveKiteFeed(
        instrument_tokens=[1],
        symbol_map={1: "AAA"},
        ticker=ticker,
    )
    await feed.connect()
    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=IST)
    ticker.push_ticks(
        [
            _tick(99, base, 100.0),  # not in symbol_map
            _tick(99, base + timedelta(minutes=1), 101.0),
        ]
    )
    assert feed.to_dataframe().empty


def test_invalid_timeframe_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        LiveKiteFeed(timeframe="3m")


def test_protocol_matches_mock() -> None:
    ticker: TickerProtocol = MockKiteTicker()
    assert hasattr(ticker, "subscribe")
