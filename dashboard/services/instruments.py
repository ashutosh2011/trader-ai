"""Kite-sourced NSE instruments cache + search.

The dashboard's symbol picker is backed by the rows of the
``instruments`` DuckDB table populated from
:py:meth:`kiteconnect.KiteConnect.instruments` on demand. The search
routine is intentionally simple — exact prefix on ``tradingsymbol``
beats substring on ``tradingsymbol`` beats substring on ``name`` — and
the table is kept tiny by filtering down to the ``NSE EQ`` slice that
the backtest runner accepts.

TRADEOFF: We do not auto-refresh on a schedule; the operator clicks
"Refresh NSE instruments" once the cache is stale or empty, and the
table truncates+inserts inside a single :func:`asyncio.to_thread` call.
The cache survives restarts because it lives in DuckDB, so a fresh
boot does not trigger a Kite call until the user explicitly asks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

import duckdb
import structlog
from kiteconnect.exceptions import KiteException

from config.settings import AppSettings

logger = structlog.get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# TRADEOFF: A 24h staleness window matches Kite's daily token rotation —
# the operator sees a "stale" hint once the cache predates the most
# recent trading day. The cache is never deleted on staleness; the
# search keeps working off whatever rows are present.
STALE_AFTER = timedelta(hours=24)

LAST_REFRESH_KEY = "last_refresh"


InstrumentsFetcher = Callable[[AppSettings, str], list[dict[str, Any]]]


@dataclass(frozen=True)
class Instrument:
    """One Kite-sourced instrument row used by the symbol picker."""

    instrument_token: int
    tradingsymbol: str
    name: str
    exchange: str
    instrument_type: str
    segment: str
    tick_size: float
    lot_size: int
    last_price: float

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for the API + templates."""
        return {
            "instrument_token": self.instrument_token,
            "tradingsymbol": self.tradingsymbol,
            "name": self.name,
            "exchange": self.exchange,
            "instrument_type": self.instrument_type,
            "segment": self.segment,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "last_price": self.last_price,
        }


def _default_instruments_fetcher(
    settings: AppSettings, exchange: str
) -> list[dict[str, Any]]:
    """Fetch raw instrument rows from Kite using the configured credentials.

    TRADEOFF: We bypass :class:`data.kite_client.KiteClient` because
    that wrapper deliberately exposes only a small subset of the Kite
    REST surface (historical data, orders, GTTs). Adding ``instruments``
    there would broaden every consumer's contract, so we instantiate
    :class:`kiteconnect.KiteConnect` here directly. The only side
    effect is a single REST call.
    """
    if not settings.kite_configured():
        msg = (
            "Kite credentials missing — set KITE_API_KEY and refresh "
            "KITE_ACCESS_TOKEN before refreshing instruments."
        )
        raise ValueError(msg)
    from kiteconnect import KiteConnect

    api_key = settings.kite.api_key or ""
    access_token = settings.kite.access_token or ""
    client: Any = KiteConnect(api_key=api_key)
    client.set_access_token(access_token)
    raw = client.instruments(exchange)
    if raw is None:
        return []
    return cast(list[dict[str, Any]], list(raw))


class InstrumentsService:
    """DuckDB-backed cache of NSE instruments fetched from Kite."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        settings: AppSettings,
        fetcher: InstrumentsFetcher | None = None,
    ) -> None:
        """Construct the service bound to a DuckDB connection.

        Args:
            conn: DuckDB connection that owns the ``instruments``
                table; the connection must already exist.
            settings: Application settings (read for Kite credentials).
            fetcher: Override the default Kite-backed fetch (used by
                tests). Receives ``(settings, exchange)`` and returns
                a list of raw instrument dicts shaped like
                :py:meth:`KiteConnect.instruments`.
        """
        self._conn = conn
        self._settings = settings
        self._fetcher = fetcher or _default_instruments_fetcher

    def ensure_schema(self) -> None:
        """Create the ``instruments`` + ``instruments_meta`` tables if missing."""
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS instruments ("
            "instrument_token BIGINT PRIMARY KEY,"
            "tradingsymbol VARCHAR NOT NULL,"
            "name VARCHAR NOT NULL,"
            "exchange VARCHAR NOT NULL,"
            "instrument_type VARCHAR NOT NULL,"
            "segment VARCHAR NOT NULL,"
            "tick_size DOUBLE NOT NULL,"
            "lot_size INTEGER NOT NULL,"
            "last_price DOUBLE NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS instruments_meta ("
            "key VARCHAR PRIMARY KEY,"
            "value VARCHAR NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS instruments_symbol_idx "
            "ON instruments(tradingsymbol)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS instruments_exchange_type_idx "
            "ON instruments(exchange, instrument_type)"
        )

    def status(self) -> dict[str, Any]:
        """Return a snapshot the UI shows beside the picker."""
        row = self._conn.execute("SELECT COUNT(*) FROM instruments").fetchone()
        count = int(row[0]) if row else 0
        last_refresh_iso = self._read_meta(LAST_REFRESH_KEY)
        stale = True
        last_refresh: str | None = None
        if last_refresh_iso:
            try:
                refreshed_at = datetime.fromisoformat(last_refresh_iso)
            except ValueError:
                refreshed_at = None
            if refreshed_at is not None:
                last_refresh = refreshed_at.isoformat()
                stale = (datetime.now(tz=IST) - refreshed_at) > STALE_AFTER
        return {
            "row_count": count,
            "last_refresh": last_refresh,
            "kite_configured": self._settings.kite_configured(),
            "stale": stale,
        }

    def refresh(
        self,
        *,
        exchange: str = "NSE",
        types: tuple[str, ...] = ("EQ",),
    ) -> int:
        """Fetch and replace the instruments cache.

        Args:
            exchange: Kite exchange code (``"NSE"`` covers the cash
                segment we backtest).
            types: Allowed ``instrument_type`` values; rows with other
                types are filtered out.

        Returns:
            The number of rows inserted.

        Raises:
            ValueError: When Kite credentials are missing or Kite raises
                a :class:`KiteException` (typically a stale token).
        """
        try:
            raw_rows = self._fetcher(self._settings, exchange)
        except KiteException as exc:
            msg = self._kite_refresh_error_message(exc)
            raise ValueError(msg) from exc

        accepted_types = set(types)
        rows: list[Instrument] = []
        for entry in raw_rows:
            instrument_type = str(entry.get("instrument_type", ""))
            if accepted_types and instrument_type not in accepted_types:
                continue
            try:
                token = int(entry.get("instrument_token") or 0)
            except (TypeError, ValueError):
                continue
            if token <= 0:
                continue
            tradingsymbol = str(entry.get("tradingsymbol") or "")
            if not tradingsymbol:
                continue
            rows.append(
                Instrument(
                    instrument_token=token,
                    tradingsymbol=tradingsymbol,
                    name=str(entry.get("name") or ""),
                    exchange=str(entry.get("exchange") or exchange),
                    instrument_type=instrument_type,
                    segment=str(entry.get("segment") or ""),
                    tick_size=float(entry.get("tick_size") or 0.0),
                    lot_size=int(entry.get("lot_size") or 0),
                    last_price=float(entry.get("last_price") or 0.0),
                )
            )

        # TRADEOFF: We replace the cache wholesale rather than upsert.
        # Kite's instrument list rotates daily (corporate actions add /
        # remove tokens), and a clean truncate avoids stale rows
        # outliving their issuer.
        self._conn.execute("DELETE FROM instruments")
        if rows:
            payload = [
                (
                    inst.instrument_token,
                    inst.tradingsymbol,
                    inst.name,
                    inst.exchange,
                    inst.instrument_type,
                    inst.segment,
                    inst.tick_size,
                    inst.lot_size,
                    inst.last_price,
                )
                for inst in rows
            ]
            self._conn.executemany(
                "INSERT INTO instruments ("
                "instrument_token, tradingsymbol, name, exchange, "
                "instrument_type, segment, tick_size, lot_size, last_price"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                payload,
            )
        self._write_meta(LAST_REFRESH_KEY, datetime.now(tz=IST).isoformat())
        logger.info(
            "dashboard_instruments_refreshed",
            exchange=exchange,
            types=list(accepted_types),
            count=len(rows),
        )
        return len(rows)

    def search(self, q: str, *, limit: int = 20) -> list[Instrument]:
        """Return ranked instrument matches for ``q``.

        Ranking: prefix on ``tradingsymbol`` first, then substring on
        ``tradingsymbol``, then substring on ``name``. Empty ``q``
        returns the first ``limit`` rows by tradingsymbol ascending.
        """
        if limit <= 0:
            return []
        target = q.strip().upper()
        if not target:
            rows = self._conn.execute(
                "SELECT instrument_token, tradingsymbol, name, exchange, "
                "instrument_type, segment, tick_size, lot_size, last_price "
                "FROM instruments ORDER BY tradingsymbol ASC LIMIT ?",
                [int(limit)],
            ).fetchall()
            return [_row_to_instrument(row) for row in rows]
        # We pull a generous pool from DuckDB then rank in Python so the
        # ranking rules stay readable. The pool is bounded to ``limit *
        # 5`` so even a single-letter query stays cheap.
        pool_size = max(int(limit) * 5, 100)
        like_value = f"%{target}%"
        rows = self._conn.execute(
            "SELECT instrument_token, tradingsymbol, name, exchange, "
            "instrument_type, segment, tick_size, lot_size, last_price "
            "FROM instruments "
            "WHERE UPPER(tradingsymbol) LIKE ? OR UPPER(name) LIKE ? "
            "ORDER BY tradingsymbol ASC LIMIT ?",
            [like_value, like_value, pool_size],
        ).fetchall()
        candidates = [_row_to_instrument(row) for row in rows]
        prefix: list[Instrument] = []
        substring: list[Instrument] = []
        name_match: list[Instrument] = []
        for inst in candidates:
            symbol_upper = inst.tradingsymbol.upper()
            if symbol_upper.startswith(target):
                prefix.append(inst)
            elif target in symbol_upper:
                substring.append(inst)
            elif target in inst.name.upper():
                name_match.append(inst)
        merged = prefix + substring + name_match
        return merged[: int(limit)]

    def get_by_symbol(self, tradingsymbol: str) -> Instrument | None:
        """Return the row whose ``tradingsymbol`` matches case-insensitively."""
        target = tradingsymbol.strip().upper()
        if not target:
            return None
        row = self._conn.execute(
            "SELECT instrument_token, tradingsymbol, name, exchange, "
            "instrument_type, segment, tick_size, lot_size, last_price "
            "FROM instruments WHERE UPPER(tradingsymbol) = ? LIMIT 1",
            [target],
        ).fetchone()
        if row is None:
            return None
        return _row_to_instrument(row)

    def get_by_token(self, instrument_token: int) -> Instrument | None:
        """Return the row whose ``instrument_token`` exactly matches."""
        row = self._conn.execute(
            "SELECT instrument_token, tradingsymbol, name, exchange, "
            "instrument_type, segment, tick_size, lot_size, last_price "
            "FROM instruments WHERE instrument_token = ? LIMIT 1",
            [int(instrument_token)],
        ).fetchone()
        if row is None:
            return None
        return _row_to_instrument(row)

    def _read_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM instruments_meta WHERE key = ?", [key]
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def _write_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO instruments_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            [key, value],
        )

    @staticmethod
    def _kite_refresh_error_message(exc: KiteException) -> str:
        """Render a friendly ValueError message for a Kite refresh failure."""
        text = str(exc)
        if (
            "api_key" in text
            or "access_token" in text
            or "Token" in type(exc).__name__
        ):
            return (
                f"Kite rejected the instruments request: {text}. "
                "Refresh today's access token from /kite, verify the API "
                "key matches the app that generated the token, then retry."
            )
        return f"Kite instruments request failed: {text}"


def _row_to_instrument(row: tuple[Any, ...]) -> Instrument:
    return Instrument(
        instrument_token=int(row[0]),
        tradingsymbol=str(row[1]),
        name=str(row[2]),
        exchange=str(row[3]),
        instrument_type=str(row[4]),
        segment=str(row[5]),
        tick_size=float(row[6]),
        lot_size=int(row[7]),
        last_price=float(row[8]),
    )


__all__ = [
    "Instrument",
    "InstrumentsFetcher",
    "InstrumentsService",
]
