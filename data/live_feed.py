"""Live Kite ticker feed with tick→bar aggregation and replay fallback."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd
import structlog

from core.bar import Bar
from data.feed import WebsocketFeed
from data.replay_feed import ReplayFeed

if TYPE_CHECKING:
    from data.kite_client import KiteClient

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

KITE_MODE_FULL = "full"

_SUPPORTED_TIMEFRAMES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
}


class TickerProtocol(Protocol):
    """Subset of :class:`kiteconnect.KiteTicker` used by :class:`LiveKiteFeed`."""

    def connect(self, threaded: bool = False) -> None: ...

    def close(self, code: int | None = None, reason: str | None = None) -> None: ...

    def subscribe(self, tokens: list[int]) -> None: ...

    def set_mode(self, mode: Any, tokens: list[int]) -> None: ...

    def on_ticks(self, callback: Any) -> None: ...

    def on_connect(self, callback: Any) -> None: ...

    def on_close(self, callback: Any) -> None: ...

    def on_error(self, callback: Any) -> None: ...

    def on_reconnect(self, callback: Any) -> None: ...


@dataclass
class PartialBar:
    """Open OHLCV bucket aggregated from ticks."""

    start_ts: datetime
    end_ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_bar(self) -> Bar:
        """Materialize the bucket as a closed :class:`Bar`."""
        return Bar(
            timestamp=self.start_ts,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


def _timeframe_to_minutes(timeframe: str) -> int:
    if timeframe not in _SUPPORTED_TIMEFRAMES:
        msg = (
            f"unsupported timeframe {timeframe!r}; "
            f"expected one of {sorted(_SUPPORTED_TIMEFRAMES)}"
        )
        raise ValueError(msg)
    return _SUPPORTED_TIMEFRAMES[timeframe]


def _bucket_start(ts: datetime, minutes: int) -> datetime:
    """Floor ``ts`` to the start of its ``minutes``-wide bucket (IST)."""
    local = ts.astimezone(IST)
    floored_minute = (local.minute // minutes) * minutes
    return local.replace(minute=floored_minute, second=0, microsecond=0)


def _coerce_tick_ts(value: Any) -> datetime:
    """Accept either ``datetime`` or epoch-seconds tick timestamps."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=IST)
        return value.astimezone(IST)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=IST)
    msg = f"unsupported tick timestamp type: {type(value).__name__}"
    raise ValueError(msg)


class LiveKiteFeed(WebsocketFeed):
    """Live tick → bar aggregator using ``KiteTicker``.

    TRADEOFF: Aggregates 1m/5m/15m bars from the tick stream client-side.
    Kite also serves historical candles with a 1-minute publish lag; live
    aggregation is fresher at the cost of holding the partial bucket in
    memory and re-emitting on disconnect/reconnect. For 5m/15m we floor
    each tick's IST timestamp to the bucket start.

    The same class supports a replay fallback (``replay_source=df``) used
    by the dry-run paths in the CLI and historical CI tests; in that mode
    no real ticker is built.
    """

    def __init__(
        self,
        kite_client: KiteClient | None = None,
        instrument_tokens: list[int] | None = None,
        symbol_map: dict[int, str] | None = None,
        timeframe: str = "1m",
        *,
        ticker: TickerProtocol | None = None,
        ticker_factory: Callable[[str, str], TickerProtocol] | None = None,
        replay_source: pd.DataFrame | Path | str | None = None,
        # Legacy keyword arguments retained for backward compatibility with
        # Week 5 tests; new code should pass ``kite_client``.
        api_key: str | None = None,
        access_token: str | None = None,
    ) -> None:
        self._kite_client = kite_client
        self._tokens = list(instrument_tokens or [])
        self._symbol_map = dict(symbol_map or {})
        self._timeframe = timeframe
        self._bucket_minutes = _timeframe_to_minutes(timeframe)
        self._ticker_factory = ticker_factory
        self._ticker: TickerProtocol | None = ticker
        self._replay = ReplayFeed(replay_source) if replay_source is not None else None
        self._legacy_api_key = api_key
        self._legacy_access_token = access_token

        self._partials: dict[str, PartialBar] = {}
        self._completed: deque[tuple[str, Bar]] = deque()
        self._queue: asyncio.Queue[tuple[str, Bar]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected = False

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @property
    def symbol_map(self) -> dict[int, str]:
        return dict(self._symbol_map)

    def bars(self) -> Iterator[Bar]:
        """Yield bars completed so far (replay frame or aggregated buckets)."""
        if self._replay is not None:
            yield from self._replay.bars()
            return
        for _, bar in list(self._completed):
            yield bar

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize completed bars as an OHLCV DataFrame.

        Returns:
            The replay frame when ``replay_source`` was supplied, otherwise
            a frame built from all bars completed since connect.
        """
        if self._replay is not None:
            return self._replay.to_dataframe()
        rows = [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "symbol": symbol,
            }
            for symbol, bar in self._completed
        ]
        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            )
        return pd.DataFrame(rows)

    async def connect(self) -> None:
        """Open websocket connection and subscribe to instruments.

        In replay mode this is a no-op; otherwise constructs (or reuses)
        the ticker, wires callbacks, subscribes to the configured tokens
        in MODE_FULL, and starts the connection on a worker thread.
        """
        self._loop = asyncio.get_running_loop()
        if self._replay is not None:
            self._connected = True
            logger.info("live_feed_replay_mode")
            return

        ticker = self._build_ticker()
        self._ticker = ticker
        ticker.on_ticks(self._on_ticks_callback)
        ticker.on_connect(self._on_connect_callback)
        ticker.on_close(self._on_close_callback)
        ticker.on_error(self._on_error_callback)
        ticker.on_reconnect(self._on_reconnect_callback)

        await asyncio.to_thread(ticker.connect, False)
        self._connected = True
        logger.info(
            "live_feed_connected",
            tokens=len(self._tokens),
            timeframe=self._timeframe,
        )

    async def disconnect(self) -> None:
        """Close websocket session and flush any open partial bars."""
        if self._replay is None and self._ticker is not None:
            await asyncio.to_thread(self._ticker.close)
        self._connected = False
        logger.info("live_feed_disconnected")

    def is_connected(self) -> bool:
        return self._connected

    async def stream_bars(self) -> AsyncIterator[tuple[str, Bar]]:
        """Yield ``(symbol, Bar)`` pairs as buckets close.

        TRADEOFF: This consumer-side iterator never returns; callers must
        cancel the task or close the feed to stop iteration.
        """
        while True:
            item = await self._queue.get()
            yield item

    def ingest_ticks(self, ticks: list[dict[str, Any]]) -> list[tuple[str, Bar]]:
        """Push ticks into the aggregator.

        Used by both the real ``on_ticks`` callback and tests. Returns the
        list of ``(symbol, Bar)`` pairs emitted as bucket boundaries close.

        Args:
            ticks: List of Kite-style tick dicts. Each must include
                ``instrument_token``, ``last_price``, and ``timestamp``
                (datetime or epoch seconds). Optional fields:
                ``last_traded_quantity`` for per-tick volume delta,
                ``volume`` for cumulative day volume.

        Returns:
            ``(symbol, Bar)`` tuples for every bucket closed by these ticks.
        """
        emitted: list[tuple[str, Bar]] = []
        for tick in ticks:
            closed = self._handle_tick(tick)
            if closed is not None:
                emitted.append(closed)
        return emitted

    def _handle_tick(self, tick: dict[str, Any]) -> tuple[str, Bar] | None:
        token = tick.get("instrument_token")
        if not isinstance(token, int):
            return None
        symbol = self._symbol_map.get(token)
        if symbol is None:
            # Drop ticks for instruments we did not subscribe to logically.
            return None
        try:
            ts = _coerce_tick_ts(tick.get("timestamp"))
        except ValueError:
            return None
        last_price = float(tick.get("last_price", 0.0))
        if last_price <= 0:
            return None
        bucket = _bucket_start(ts, self._bucket_minutes)
        bucket_end = bucket + timedelta(minutes=self._bucket_minutes)
        volume_delta = float(tick.get("last_traded_quantity", 0))
        cumulative_volume = tick.get("volume")

        partial = self._partials.get(symbol)
        closed: tuple[str, Bar] | None = None
        if partial is None:
            self._partials[symbol] = PartialBar(
                start_ts=bucket,
                end_ts=bucket_end,
                open=last_price,
                high=last_price,
                low=last_price,
                close=last_price,
                volume=volume_delta
                if volume_delta > 0
                else float(cumulative_volume or 0.0),
            )
            return None
        if bucket != partial.start_ts:
            closed = (symbol, partial.to_bar())
            self._completed.append(closed)
            self._enqueue(closed)
            self._partials[symbol] = PartialBar(
                start_ts=bucket,
                end_ts=bucket_end,
                open=last_price,
                high=last_price,
                low=last_price,
                close=last_price,
                volume=volume_delta
                if volume_delta > 0
                else float(cumulative_volume or 0.0),
            )
            return closed
        partial.high = max(partial.high, last_price)
        partial.low = min(partial.low, last_price)
        partial.close = last_price
        if volume_delta > 0:
            partial.volume += volume_delta
        elif cumulative_volume is not None:
            partial.volume = float(cumulative_volume)
        return None

    def _enqueue(self, item: tuple[str, Bar]) -> None:
        if self._loop is None or not self._loop.is_running():
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    # Ticker callbacks -------------------------------------------------

    def _on_ticks_callback(self, _ws: object, ticks: list[dict[str, Any]]) -> None:
        for tick in ticks:
            self._handle_tick(tick)

    def _on_connect_callback(self, ws: object, _response: Any) -> None:
        if self._tokens and self._ticker is not None:
            try:
                self._ticker.subscribe(self._tokens)
                self._ticker.set_mode(KITE_MODE_FULL, self._tokens)
            except Exception as exc:
                logger.exception("live_feed_subscribe_failed", error=str(exc))
        else:
            del ws
        logger.info("live_feed_ws_connected")

    def _on_close_callback(self, _ws: object, code: Any, reason: Any) -> None:
        logger.warning("live_feed_ws_closed", code=code, reason=reason)
        self._connected = False

    def _on_error_callback(self, _ws: object, code: Any, reason: Any) -> None:
        logger.warning("live_feed_ws_error", code=code, reason=reason)

    def _on_reconnect_callback(self, _ws: object, attempts: Any) -> None:
        logger.info("live_feed_ws_reconnect", attempts=attempts)

    def _build_ticker(self) -> TickerProtocol:
        if self._ticker is not None:
            return self._ticker
        api_key, access_token = self._resolve_credentials()
        if self._ticker_factory is not None:
            return self._ticker_factory(api_key, access_token)
        return _default_ticker_factory(api_key, access_token)

    def _resolve_credentials(self) -> tuple[str, str]:
        if self._kite_client is not None:
            session = self._kite_client.session
            return session.api_key, session.access_token
        if self._legacy_api_key and self._legacy_access_token:
            return self._legacy_api_key, self._legacy_access_token
        msg = "kite_client or api_key/access_token required for live websocket"
        raise ValueError(msg)


class LiveFeed(LiveKiteFeed):
    """Replay-only alias retained for backward compatibility."""

    def __init__(self, source: pd.DataFrame | Path | str) -> None:
        super().__init__(replay_source=source)


def _default_ticker_factory(api_key: str, access_token: str) -> TickerProtocol:
    from kiteconnect import KiteTicker

    return cast(TickerProtocol, KiteTicker(api_key, access_token))
