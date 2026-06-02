"""Persistent state for the GTT-based execution lifecycle.

The store backs :class:`execution.kite.KiteBroker` so the broker can survive
process restarts and behave idempotently across them. We persist one row per
``client_order_id`` (the :func:`execution.broker.deterministic_client_order_id`
hash) and walk it through a small state machine:

``PENDING_ENTRY → ENTERED → EXITED``

with terminal branches ``FAILED`` (entry was rejected / never filled) and
``CANCELLED`` (manual ``flatten_all`` / kill-switch recovery).

TRADEOFF: We use DuckDB instead of SQLite to match the rest of the data
layer (``data/store.py``) — one engine, one set of file-locking semantics.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import duckdb
import structlog
from pydantic import BaseModel, ConfigDict, Field

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)


class OrderState(StrEnum):
    """Lifecycle states for a tracked bracket-via-GTT order."""

    PENDING_ENTRY = "PENDING_ENTRY"
    ENTERED = "ENTERED"
    EXITED = "EXITED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


OPEN_STATES: frozenset[OrderState] = frozenset(
    {OrderState.PENDING_ENTRY, OrderState.ENTERED}
)


class OrderRecord(BaseModel):
    """One persisted bracket-via-GTT order with full lifecycle metadata."""

    model_config = ConfigDict(frozen=False)

    client_order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    target: float = Field(gt=0)
    state: OrderState
    entry_order_id: str | None = None
    sl_gtt_id: int | None = None
    target_gtt_id: int | None = None
    fill_price: float | None = None
    exit_price: float | None = None
    pnl: float | None = None
    signal_ts: datetime
    created_at: datetime
    updated_at: datetime
    strategy_id: str
    error: str | None = None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS order_records (
    client_order_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    qty INTEGER NOT NULL,
    entry_price DOUBLE NOT NULL,
    stop_loss DOUBLE NOT NULL,
    target DOUBLE NOT NULL,
    state VARCHAR NOT NULL,
    entry_order_id VARCHAR,
    sl_gtt_id BIGINT,
    target_gtt_id BIGINT,
    fill_price DOUBLE,
    exit_price DOUBLE,
    pnl DOUBLE,
    signal_ts TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    strategy_id VARCHAR NOT NULL,
    error VARCHAR
);
"""

_COLUMNS: tuple[str, ...] = (
    "client_order_id",
    "symbol",
    "side",
    "qty",
    "entry_price",
    "stop_loss",
    "target",
    "state",
    "entry_order_id",
    "sl_gtt_id",
    "target_gtt_id",
    "fill_price",
    "exit_price",
    "pnl",
    "signal_ts",
    "created_at",
    "updated_at",
    "strategy_id",
    "error",
)


class OrderStateStore:
    """DuckDB-backed persistence for :class:`OrderRecord` rows.

    The store is intentionally tiny — upsert/get/list — because the broker
    owns all state-machine semantics and only delegates persistence here.

    TRADEOFF: The DuckDB file must be backed up alongside the candle store.
    Loss of this file means the broker can no longer reconcile its in-flight
    GTTs back to a strategy on restart, and manual flatten is the only safe
    recovery path. We accept that cost in exchange for a single-file,
    zero-server persistence layer.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._path))
        self._conn.execute(SCHEMA_SQL)
        logger.info("order_state_store_opened", path=str(self._path))

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()

    def upsert(self, record: OrderRecord) -> None:
        """Insert or update one record by ``client_order_id``."""
        payload = _record_to_row(record)
        placeholders = ", ".join(["?"] * len(_COLUMNS))
        updates = ", ".join(
            f"{col} = excluded.{col}" for col in _COLUMNS if col != "client_order_id"
        )
        sql = (
            f"INSERT INTO order_records ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (client_order_id) DO UPDATE SET {updates}"
        )
        self._conn.execute(sql, payload)
        logger.info(
            "order_state_upsert",
            client_order_id=record.client_order_id,
            state=record.state.value,
            symbol=record.symbol,
        )

    def get(self, client_order_id: str) -> OrderRecord | None:
        """Return the record for ``client_order_id``, or ``None`` if missing."""
        cur = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM order_records "
            "WHERE client_order_id = ?",
            [client_order_id],
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def list_open(self) -> list[OrderRecord]:
        """Return every record in ``PENDING_ENTRY`` or ``ENTERED`` state."""
        open_states = [s.value for s in OPEN_STATES]
        placeholders = ",".join(["?"] * len(open_states))
        cur = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM order_records "
            f"WHERE state IN ({placeholders}) "
            "ORDER BY created_at",
            open_states,
        )
        return [_row_to_record(row) for row in cur.fetchall()]

    def list_all(self) -> list[OrderRecord]:
        """Return every record in the store, oldest first."""
        cur = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM order_records ORDER BY created_at"
        )
        return [_row_to_record(row) for row in cur.fetchall()]


def _record_to_row(record: OrderRecord) -> list[object]:
    return [
        record.client_order_id,
        record.symbol,
        record.side,
        record.qty,
        record.entry_price,
        record.stop_loss,
        record.target,
        record.state.value,
        record.entry_order_id,
        record.sl_gtt_id,
        record.target_gtt_id,
        record.fill_price,
        record.exit_price,
        record.pnl,
        _to_ist(record.signal_ts),
        _to_ist(record.created_at),
        _to_ist(record.updated_at),
        record.strategy_id,
        record.error,
    ]


def _row_to_record(row: tuple[object, ...]) -> OrderRecord:
    mapping = dict(zip(_COLUMNS, row, strict=True))
    return OrderRecord(
        client_order_id=str(mapping["client_order_id"]),
        symbol=str(mapping["symbol"]),
        side="BUY" if str(mapping["side"]) == "BUY" else "SELL",
        qty=int(_as_int(mapping["qty"])),
        entry_price=float(_as_float(mapping["entry_price"])),
        stop_loss=float(_as_float(mapping["stop_loss"])),
        target=float(_as_float(mapping["target"])),
        state=OrderState(str(mapping["state"])),
        entry_order_id=_opt_str(mapping["entry_order_id"]),
        sl_gtt_id=_opt_int(mapping["sl_gtt_id"]),
        target_gtt_id=_opt_int(mapping["target_gtt_id"]),
        fill_price=_opt_float(mapping["fill_price"]),
        exit_price=_opt_float(mapping["exit_price"]),
        pnl=_opt_float(mapping["pnl"]),
        signal_ts=_as_ist_datetime(mapping["signal_ts"]),
        created_at=_as_ist_datetime(mapping["created_at"]),
        updated_at=_as_ist_datetime(mapping["updated_at"]),
        strategy_id=str(mapping["strategy_id"]),
        error=_opt_str(mapping["error"]),
    )


def _to_ist(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=IST)
    return ts.astimezone(IST)


def _as_ist_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        msg = f"expected datetime, got {type(value).__name__}"
        raise TypeError(msg)
    return _to_ist(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    msg = f"expected int, got {type(value).__name__}"
    raise TypeError(msg)


def _as_float(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    msg = f"expected number, got {type(value).__name__}"
    raise TypeError(msg)


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _opt_int(value: object) -> int | None:
    if value is None:
        return None
    return _as_int(value)


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    return _as_float(value)
