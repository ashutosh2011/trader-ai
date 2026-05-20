"""Live Kite ticker feed with replay fallback for dev/CI."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd
import structlog

from core.bar import Bar
from data.feed import WebsocketFeed
from data.replay_feed import ReplayFeed

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)


class TickerProtocol(Protocol):
    """Subset of kiteconnect KiteTicker used by :class:`LiveKiteFeed`."""

    def connect(self, threaded: bool = False) -> None: ...

    def close(self) -> None: ...

    def subscribe(self, tokens: list[int]) -> None: ...

    def set_mode(self, mode: Any, tokens: list[int]) -> None: ...

    def on_ticks(self, callback: Any) -> None: ...


def _tick_to_bar(tick: dict[str, Any]) -> Bar | None:
    """Convert a Kite tick dict to a Bar when OHLC is present."""
    if "ohlc" not in tick:
        return None
    ohlc = tick["ohlc"]
    ts = datetime.fromtimestamp(tick["timestamp"], tz=IST)
    return Bar(
        timestamp=ts,
        open=float(ohlc["open"]),
        high=float(ohlc["high"]),
        low=float(ohlc["low"]),
        close=float(tick.get("last_price", ohlc["close"])),
        volume=float(tick.get("volume", 0)),
    )


class LiveKiteFeed(WebsocketFeed):
    """Kite ticker websocket feed; falls back to replay when credentials absent.

    TRADEOFF: Buffers ticks in-memory for ``bars()`` iteration; production
    deployments should drain via async callbacks instead of materializing all bars.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        access_token: str | None = None,
        instrument_tokens: list[int] | None = None,
        replay_source: pd.DataFrame | Path | str | None = None,
        ticker: TickerProtocol | None = None,
    ) -> None:
        self._api_key = api_key
        self._access_token = access_token
        self._tokens = instrument_tokens or []
        self._replay = ReplayFeed(replay_source) if replay_source is not None else None
        self._ticker = ticker
        self._buffer: deque[Bar] = deque()
        self._frame: pd.DataFrame | None = None
        self._connected = False

    def bars(self) -> Iterator[Bar]:
        if self._replay is not None:
            yield from self._replay.bars()
            return
        yield from self._buffer

    def to_dataframe(self) -> pd.DataFrame:
        if self._replay is not None:
            return self._replay.to_dataframe()
        if self._frame is not None:
            return self._frame.copy()
        rows = [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in self._buffer
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    async def connect(self) -> None:
        """Connect websocket or validate replay fallback."""
        if self._replay is not None:
            self._connected = True
            logger.info("live_feed_replay_mode")
            return
        if not self._api_key or not self._access_token:
            msg = "api_key and access_token required for live websocket"
            raise ValueError(msg)
        ticker = self._ticker or _build_ticker(self._api_key, self._access_token)
        self._ticker = ticker

        def on_ticks(ws: object, ticks: list[dict[str, Any]]) -> None:
            for tick in ticks:
                bar = _tick_to_bar(tick)
                if bar is not None:
                    self._buffer.append(bar)

        ticker.on_ticks(on_ticks)
        await asyncio.to_thread(ticker.connect, False)
        if self._tokens:
            await asyncio.to_thread(ticker.subscribe, self._tokens)
            await asyncio.to_thread(ticker.set_mode, "full", self._tokens)
        self._connected = True
        logger.info("live_feed_connected", tokens=len(self._tokens))

    async def disconnect(self) -> None:
        """Close websocket session."""
        if self._ticker is not None and self._replay is None:
            await asyncio.to_thread(self._ticker.close)
        self._connected = False
        logger.info("live_feed_disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def ingest_ticks(self, ticks: list[dict[str, Any]]) -> None:
        """Push ticks into the buffer (used by tests without a real socket)."""
        for tick in ticks:
            bar = _tick_to_bar(tick)
            if bar is not None:
                self._buffer.append(bar)


class LiveFeed(LiveKiteFeed):
    """Alias retained for backward compatibility with Week 5 imports."""

    def __init__(self, source: pd.DataFrame | Path | str) -> None:
        super().__init__(replay_source=source)


def _build_ticker(api_key: str, access_token: str) -> TickerProtocol:
    from kiteconnect import KiteTicker

    return cast(TickerProtocol, KiteTicker(api_key, access_token))
