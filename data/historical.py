"""Bulk historical candle fetch with gap detection and DuckDB persistence."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from pydantic import BaseModel, ConfigDict

from data.kite_client import KiteClient
from data.store import CandleStore

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

INTERVAL_MINUTES: dict[str, int] = {
    "minute": 1,
    "3minute": 3,
    "5minute": 5,
    "10minute": 10,
    "15minute": 15,
    "30minute": 30,
    "60minute": 60,
    "day": 375,
}


class GapRange(BaseModel):
    """Detected missing candle interval."""

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime


class HistoricalSyncResult(BaseModel):
    """Outcome of a historical fetch + store operation."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: str
    rows_fetched: int
    rows_stored: int
    gaps_filled: int


def kite_candles_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert Kite historical API rows to OHLCV DataFrame."""
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={"date": "timestamp"})
    ts = pd.to_datetime(frame["timestamp"])
    ts = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
    frame["timestamp"] = ts
    return frame[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def detect_gaps(
    bars: pd.DataFrame,
    timeframe: str,
    *,
    tolerance_factor: float = 1.5,
) -> list[GapRange]:
    """Detect missing intervals in sorted OHLCV data."""
    if len(bars) < 2:
        return []
    step = INTERVAL_MINUTES.get(timeframe)
    if step is None:
        return []
    expected = timedelta(minutes=step)
    gaps: list[GapRange] = []
    timestamps = pd.to_datetime(bars["timestamp"])
    for idx in range(len(timestamps) - 1):
        delta = timestamps.iloc[idx + 1] - timestamps.iloc[idx]
        if delta > expected * tolerance_factor:
            gaps.append(
                GapRange(
                    start=timestamps.iloc[idx].to_pydatetime(),
                    end=timestamps.iloc[idx + 1].to_pydatetime(),
                )
            )
    return gaps


class HistoricalFetcher:
    """Fetch historical candles via Kite and persist to DuckDB."""

    def __init__(self, client: KiteClient, store: CandleStore) -> None:
        self._client = client
        self._store = store

    def fetch_and_store(
        self,
        *,
        symbol: str,
        instrument_token: int,
        timeframe: str,
        from_date: datetime,
        to_date: datetime,
        fill_gaps: bool = True,
    ) -> HistoricalSyncResult:
        """Fetch candles, optionally fill gaps, and write to the store."""
        rows = self._client.historical_data(
            instrument_token,
            from_date.astimezone(IST),
            to_date.astimezone(IST),
            timeframe,
        )
        frame = kite_candles_to_dataframe(rows)
        gaps_filled = 0
        if fill_gaps and not frame.empty:
            for gap in detect_gaps(frame, timeframe):
                patch_rows = self._client.historical_data(
                    instrument_token,
                    gap.start,
                    gap.end,
                    timeframe,
                )
                patch = kite_candles_to_dataframe(patch_rows)
                if not patch.empty:
                    frame = (
                        pd.concat([frame, patch], ignore_index=True)
                        .drop_duplicates(subset=["timestamp"])
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )
                    gaps_filled += 1
        stored = self._store.upsert_bars(symbol, timeframe, frame)
        logger.info(
            "historical_sync_complete",
            symbol=symbol,
            timeframe=timeframe,
            rows=len(frame),
            gaps_filled=gaps_filled,
        )
        return HistoricalSyncResult(
            symbol=symbol,
            timeframe=timeframe,
            rows_fetched=len(frame),
            rows_stored=stored,
            gaps_filled=gaps_filled,
        )
