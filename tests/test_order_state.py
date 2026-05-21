"""Tests for the persistent order state store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from execution.order_state import OrderRecord, OrderState, OrderStateStore

IST = ZoneInfo("Asia/Kolkata")


def _record(
    *,
    client_order_id: str = "tb-aaaa",
    symbol: str = "RELIANCE",
    state: OrderState = OrderState.PENDING_ENTRY,
) -> OrderRecord:
    now = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    return OrderRecord(
        client_order_id=client_order_id,
        symbol=symbol,
        side="BUY",
        qty=10,
        entry_price=2500.0,
        stop_loss=2480.0,
        target=2540.0,
        state=state,
        entry_order_id="241201000000001",
        signal_ts=now,
        created_at=now,
        updated_at=now,
        strategy_id="ema_crossover",
    )


def test_store_roundtrip(tmp_path: Path) -> None:
    store = OrderStateStore(tmp_path / "state.duckdb")
    try:
        record = _record()
        store.upsert(record)
        fetched = store.get(record.client_order_id)
        assert fetched is not None
        assert fetched.client_order_id == record.client_order_id
        assert fetched.symbol == record.symbol
        assert fetched.state == OrderState.PENDING_ENTRY
        assert fetched.entry_order_id == record.entry_order_id
        assert fetched.signal_ts.tzinfo is not None
    finally:
        store.close()


def test_store_missing_get_returns_none(tmp_path: Path) -> None:
    store = OrderStateStore(tmp_path / "state.duckdb")
    try:
        assert store.get("does-not-exist") is None
    finally:
        store.close()


def test_store_upsert_overwrites(tmp_path: Path) -> None:
    store = OrderStateStore(tmp_path / "state.duckdb")
    try:
        record = _record()
        store.upsert(record)
        updated = record.model_copy(
            update={"state": OrderState.ENTERED, "fill_price": 2501.0}
        )
        store.upsert(updated)
        fetched = store.get(record.client_order_id)
        assert fetched is not None
        assert fetched.state == OrderState.ENTERED
        assert fetched.fill_price == 2501.0
        assert len(store.list_all()) == 1
    finally:
        store.close()


def test_list_open_filters_terminal_states(tmp_path: Path) -> None:
    store = OrderStateStore(tmp_path / "state.duckdb")
    try:
        store.upsert(_record(client_order_id="tb-a", state=OrderState.PENDING_ENTRY))
        store.upsert(_record(client_order_id="tb-b", state=OrderState.ENTERED))
        store.upsert(_record(client_order_id="tb-c", state=OrderState.EXITED))
        store.upsert(_record(client_order_id="tb-d", state=OrderState.FAILED))
        store.upsert(_record(client_order_id="tb-e", state=OrderState.CANCELLED))
        open_records = store.list_open()
        ids = {r.client_order_id for r in open_records}
        assert ids == {"tb-a", "tb-b"}
        assert len(store.list_all()) == 5
    finally:
        store.close()


def test_store_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "state.duckdb"
    store = OrderStateStore(db_path)
    store.upsert(_record())
    store.close()

    reopened = OrderStateStore(db_path)
    try:
        fetched = reopened.get("tb-aaaa")
        assert fetched is not None
        assert fetched.state == OrderState.PENDING_ENTRY
    finally:
        reopened.close()
