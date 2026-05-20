"""DuckDB OHLCV candle store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import structlog
from pydantic import BaseModel, ConfigDict

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS candles (
    symbol VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
"""


class CandleKey(BaseModel):
    """Lookup key for stored candles."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: str


class CandleStore:
    """Persist and retrieve OHLCV bars by symbol and timeframe."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._path))
        self._conn.execute(SCHEMA_SQL)
        logger.info("candle_store_opened", path=str(self._path))

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

    def upsert_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> int:
        """Insert or replace OHLCV rows for ``symbol``/``timeframe``.

        Returns:
            Number of rows written.
        """
        frame = _normalize_bars(bars)
        if frame.empty:
            return 0
        frame = frame.copy()
        frame["symbol"] = symbol
        frame["timeframe"] = timeframe
        self._conn.register("_batch", frame)
        self._conn.execute(
            """
            INSERT INTO candles
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume
            FROM _batch
            ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
            """
        )
        self._conn.unregister("_batch")
        count = len(frame)
        logger.info("candles_upserted", symbol=symbol, timeframe=timeframe, rows=count)
        return count

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV bars for a symbol/timeframe, optionally filtered by time."""
        clauses = ["symbol = ?", "timeframe = ?"]
        params: list[object] = [symbol, timeframe]
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(_to_ist(start))
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(_to_ist(end))
        where = " AND ".join(clauses)
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE {where}
            ORDER BY timestamp
        """
        frame = self._conn.execute(query, params).fetchdf()
        if frame.empty:
            return _empty_ohlcv()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(IST)
        return frame.reset_index(drop=True)

    def latest_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        """Return the newest stored bar timestamp, if any."""
        row = self._conn.execute(
            """
            SELECT MAX(timestamp) FROM candles
            WHERE symbol = ? AND timeframe = ?
            """,
            [symbol, timeframe],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        ts = pd.Timestamp(row[0])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime().astimezone(IST)

    def symbols(self) -> list[CandleKey]:
        """List symbol/timeframe pairs present in the store."""
        rows = self._conn.execute(
            "SELECT DISTINCT symbol, timeframe FROM candles ORDER BY 1, 2"
        ).fetchall()
        return [CandleKey(symbol=str(r[0]), timeframe=str(r[1])) for r in rows]


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        msg = f"bars missing columns: {sorted(missing)}"
        raise ValueError(msg)
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(IST)
    return frame.reset_index(drop=True)


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


def _to_ist(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=IST)
    return ts.astimezone(IST)
