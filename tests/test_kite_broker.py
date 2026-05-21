"""Tests for the GTT-based :class:`KiteBroker`.

Covers placement, polling state transitions, OCO GTT lifecycle, idempotency,
and the confirmed ``flatten_all`` close path. A small in-process
:class:`FakeKiteClient` records every call and lets each test drive Kite
responses deterministically without hitting the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from kiteconnect import exceptions as kite_exceptions

from config.settings import AppSettings
from execution.broker import FlattenIncomplete
from execution.kite import KiteBroker
from execution.order_state import OrderState, OrderStateStore
from tests.test_risk_manager import _signal

IST = ZoneInfo("Asia/Kolkata")


class FakeKiteClient:
    """In-memory KiteBrokerClient that records every call.

    Each test populates ``orders_response`` / ``gtts_response`` /
    ``positions_response`` to control what the broker observes during
    polling and replays.
    """

    def __init__(self) -> None:
        self.placed_orders: list[dict[str, Any]] = []
        self.placed_gtts: list[dict[str, Any]] = []
        self.deleted_gtts: list[int] = []
        self.orders_response: list[dict[str, Any]] = []
        self.gtts_response: list[dict[str, Any]] = []
        self.positions_response: dict[str, list[dict[str, Any]]] = {"net": []}
        self.next_order_id: str = "BROKER-1"
        self.next_trigger_id: int = 90_000

    def place_order(self, **kwargs: Any) -> str:
        self.placed_orders.append(kwargs)
        order_id = self.next_order_id
        self.next_order_id = f"BROKER-{int(self.next_order_id.split('-')[1]) + 1}"
        return order_id

    def place_gtt(self, **kwargs: Any) -> dict[str, Any]:
        self.placed_gtts.append(kwargs)
        trigger_id = self.next_trigger_id
        self.next_trigger_id += 1
        return {"trigger_id": trigger_id}

    def get_gtts(self) -> list[dict[str, Any]]:
        return self.gtts_response

    def delete_gtt(self, trigger_id: int) -> dict[str, Any]:
        self.deleted_gtts.append(trigger_id)
        return {"trigger_id": trigger_id}

    def orders(self) -> list[dict[str, Any]]:
        return self.orders_response

    def positions(self) -> dict[str, list[dict[str, Any]]]:
        return self.positions_response


@pytest.fixture
def store(tmp_path: Path) -> Iterator[OrderStateStore]:
    s = OrderStateStore(tmp_path / "state.duckdb")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def fake() -> FakeKiteClient:
    return FakeKiteClient()


def _bracket_signal(symbol: str = "RELIANCE") -> Any:
    return _signal(
        datetime(2024, 1, 1, 10, 0, tzinfo=IST),
        entry=2500.0,
        stop=2480.0,
        target=2540.0,
    ).model_copy(update={"symbol": symbol})


def _broker(
    fake: FakeKiteClient,
    store: OrderStateStore | None = None,
    *,
    product: str | None = None,
) -> KiteBroker:
    return KiteBroker(
        fake,
        settings=AppSettings(),
        state_store=store,
        product=product,
        sleep=lambda _s: None,
    )


# ---------------------------------------------------------------------------
# Basic placement / rejection / idempotency
# ---------------------------------------------------------------------------


def test_kite_bracket_order_placed() -> None:
    mock = MagicMock()
    mock.place_order.return_value = "12345"
    mock.positions.return_value = {"net": []}
    broker = KiteBroker(mock, settings=AppSettings())
    signal = _signal(datetime(2024, 1, 1, 10, 0, tzinfo=IST))
    result = broker.place_bracket_order(signal, qty=1)
    assert result.status == "PENDING"
    mock.place_order.assert_called_once()
    assert result.client_order_id.startswith("tb-")


def test_kite_rejects_zero_qty() -> None:
    mock = MagicMock()
    broker = KiteBroker(mock, settings=AppSettings())
    signal = _signal(datetime(2024, 1, 1, 10, 0, tzinfo=IST))
    result = broker.place_bracket_order(signal, qty=0)
    assert result.status == "REJECTED"
    mock.place_order.assert_not_called()


def test_kite_idempotent_duplicate() -> None:
    mock = MagicMock()
    mock.place_order.return_value = "1"
    broker = KiteBroker(mock, settings=AppSettings())
    signal = _signal(datetime(2024, 1, 1, 10, 0, tzinfo=IST))
    first = broker.place_bracket_order(signal, qty=1)
    second = broker.place_bracket_order(signal, qty=1)
    assert first.status == "PENDING"
    assert second.message == "duplicate_client_order_id"
    assert mock.place_order.call_count == 1


def test_kite_persists_pending_entry_to_store(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    broker = _broker(fake, store)
    signal = _bracket_signal()
    result = broker.place_bracket_order(signal, qty=1)

    record = store.get(result.client_order_id)
    assert record is not None
    assert record.state == OrderState.PENDING_ENTRY
    assert record.entry_order_id == "BROKER-1"
    assert record.symbol == "RELIANCE"
    # No GTT placed yet — that happens only on entry fill.
    assert fake.placed_gtts == []


def test_kite_idempotent_with_state_store(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    """Persistent store ⇒ a fresh KiteBroker instance still sees the duplicate."""
    broker = _broker(fake, store)
    signal = _bracket_signal()
    broker.place_bracket_order(signal, qty=1)
    # New broker instance using the same store — simulates process restart.
    broker_restart = _broker(fake, store)
    second = broker_restart.place_bracket_order(signal, qty=1)
    assert second.message == "duplicate_client_order_id"
    assert len(fake.placed_orders) == 1


def test_kite_bad_input_not_retried(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    """InputException is fail-fast — no retries, REJECTED result."""
    fake_with_bad = FakeKiteClient()

    def boom(**_: Any) -> str:
        raise kite_exceptions.InputException("missing qty")

    fake_with_bad.place_order = boom  # type: ignore[method-assign]
    broker = _broker(fake_with_bad, store)
    result = broker.place_bracket_order(_bracket_signal(), qty=1)
    assert result.status == "REJECTED"
    assert store.get(result.client_order_id) is None


# ---------------------------------------------------------------------------
# poll_and_advance: state machine transitions
# ---------------------------------------------------------------------------


def test_poll_pending_entry_complete_promotes_to_entered(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    broker = _broker(fake, store)
    signal = _bracket_signal()
    placed = broker.place_bracket_order(signal, qty=1)
    fake.orders_response = [
        {
            "order_id": "BROKER-1",
            "status": "COMPLETE",
            "average_price": 2501.0,
            "tradingsymbol": "RELIANCE",
            "tag": placed.client_order_id,
        }
    ]
    broker.poll_and_advance()

    record = store.get(placed.client_order_id)
    assert record is not None
    assert record.state == OrderState.ENTERED
    assert record.fill_price == 2501.0
    assert record.sl_gtt_id is not None
    # Exactly one OCO GTT (two-leg) — not two single-leg GTTs.
    assert len(fake.placed_gtts) == 1
    payload = fake.placed_gtts[0]
    assert payload["trigger_type"] == "two-leg"
    # trigger_values ascending: SL below entry, target above (BUY direction)
    assert payload["trigger_values"] == [signal.stop_loss, signal.target]
    assert len(payload["orders"]) == 2


def test_poll_pending_entry_rejected_marks_failed(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    broker = _broker(fake, store)
    placed = broker.place_bracket_order(_bracket_signal(), qty=1)
    fake.orders_response = [
        {
            "order_id": "BROKER-1",
            "status": "REJECTED",
            "status_message": "exchange_rejection",
            "tradingsymbol": "RELIANCE",
        }
    ]
    broker.poll_and_advance()
    record = store.get(placed.client_order_id)
    assert record is not None
    assert record.state == OrderState.FAILED
    assert record.error is not None
    # No GTT placed for a rejected entry.
    assert fake.placed_gtts == []


def test_poll_entered_gtt_triggered_marks_exited(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    broker = _broker(fake, store)
    placed = broker.place_bracket_order(_bracket_signal(), qty=1)
    fake.orders_response = [
        {
            "order_id": "BROKER-1",
            "status": "COMPLETE",
            "average_price": 2500.0,
            "tradingsymbol": "RELIANCE",
        }
    ]
    broker.poll_and_advance()
    record = store.get(placed.client_order_id)
    assert record is not None
    assert record.state == OrderState.ENTERED
    gtt_id = record.sl_gtt_id
    assert gtt_id is not None

    # Now the SL leg of the OCO triggers — the broker sees it via get_gtts().
    fake.orders_response = [
        {
            "order_id": "EXIT-1",
            "status": "COMPLETE",
            "average_price": 2480.0,
            "tradingsymbol": "RELIANCE",
        }
    ]
    fake.gtts_response = [
        {
            "id": gtt_id,
            "status": "triggered",
            "triggered_at_price": 2480.0,
            "result": {"order_result": {"order_id": "EXIT-1", "average_price": 2480.0}},
        }
    ]
    broker.poll_and_advance()
    record = store.get(placed.client_order_id)
    assert record is not None
    assert record.state == OrderState.EXITED
    assert record.exit_price == 2480.0
    # SL @ 2480 from entry 2500 with qty=1 → pnl = -20
    assert record.pnl == pytest.approx(-20.0)


def test_poll_no_open_records_no_calls(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    """poll_and_advance with an empty store short-circuits before API calls."""
    broker = _broker(fake, store)
    broker.poll_and_advance()
    assert fake.placed_orders == []
    assert fake.placed_gtts == []


# ---------------------------------------------------------------------------
# get_positions / state-store enrichment
# ---------------------------------------------------------------------------


def test_get_positions_enriches_levels_from_store(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    broker = _broker(fake, store)
    broker.place_bracket_order(_bracket_signal(), qty=1)
    fake.orders_response = [
        {
            "order_id": "BROKER-1",
            "status": "COMPLETE",
            "average_price": 2500.0,
            "tradingsymbol": "RELIANCE",
        }
    ]
    broker.poll_and_advance()

    fake.positions_response = {
        "net": [
            {
                "tradingsymbol": "RELIANCE",
                "quantity": 1,
                "average_price": 2500.0,
            }
        ]
    }
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].stop_loss == 2480.0
    assert positions[0].target == 2540.0


def test_get_positions_passthrough_when_no_record(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    broker = _broker(fake, store)
    fake.positions_response = {
        "net": [
            {
                "tradingsymbol": "INFY",
                "quantity": 5,
                "average_price": 1500.0,
            }
        ]
    }
    positions = broker.get_positions()
    assert positions[0].stop_loss is None
    assert positions[0].target is None
    assert positions[0].strategy_id == "kite_sync"


# ---------------------------------------------------------------------------
# flatten_all with poll-confirmation budget
# ---------------------------------------------------------------------------


def test_flatten_all_market_squareoff_with_confirmation(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    broker = _broker(fake, store)
    broker.place_bracket_order(_bracket_signal(), qty=1)
    fake.orders_response = [
        {
            "order_id": "BROKER-1",
            "status": "COMPLETE",
            "average_price": 2500.0,
            "tradingsymbol": "RELIANCE",
        }
    ]
    broker.poll_and_advance()

    fake.positions_response = {
        "net": [
            {"tradingsymbol": "RELIANCE", "quantity": 1, "average_price": 2500.0}
        ]
    }

    # After the first square-off MARKET order, positions go to zero.
    poll_count = {"n": 0}
    original_positions_response = fake.positions_response

    def positions_after_squareoff() -> dict[str, list[dict[str, Any]]]:
        poll_count["n"] += 1
        if poll_count["n"] >= 2:
            return {"net": []}
        return original_positions_response

    fake.positions = positions_after_squareoff  # type: ignore[method-assign]

    broker.flatten_all()
    # The market square-off was placed (entry + flatten).
    market_orders = [
        o for o in fake.placed_orders if o.get("order_type") == "MARKET"
    ]
    assert len(market_orders) == 1
    assert market_orders[0]["transaction_type"] == "SELL"
    # GTT was cancelled.
    assert len(fake.deleted_gtts) == 1


def test_flatten_all_exhausts_budget_raises(
    fake: FakeKiteClient,
    store: OrderStateStore,
) -> None:
    broker = _broker(fake, store)
    fake.positions_response = {
        "net": [
            {"tradingsymbol": "RELIANCE", "quantity": 1, "average_price": 2500.0}
        ]
    }
    with pytest.raises(FlattenIncomplete) as excinfo:
        broker.flatten_all()
    assert excinfo.value.open_positions[0].symbol == "RELIANCE"
    assert excinfo.value.attempts == 10


# ---------------------------------------------------------------------------
# Product resolution heuristic
# ---------------------------------------------------------------------------


def test_product_resolution_equity_defaults_to_mis(fake: FakeKiteClient) -> None:
    broker = _broker(fake)
    # INFY ends in neither FUT/CE/PE so the heuristic picks MIS.
    broker.place_bracket_order(_bracket_signal("INFY"), qty=1)
    assert fake.placed_orders[0]["product"] == "MIS"


def test_product_resolution_options_defaults_to_nrml(fake: FakeKiteClient) -> None:
    broker = _broker(fake)
    broker.place_bracket_order(_bracket_signal("NIFTY24MAY24000CE"), qty=1)
    assert fake.placed_orders[0]["product"] == "NRML"


def test_product_resolution_ctor_arg_wins(fake: FakeKiteClient) -> None:
    """Explicit ctor product overrides the symbol-suffix heuristic.

    Necessary because the heuristic has known false positives on equity
    symbols ending in ``CE`` / ``PE`` (e.g. ``RELIANCE``); when the caller
    knows better they should pass ``product=`` explicitly.
    """
    broker = _broker(fake, product="CNC")
    broker.place_bracket_order(_bracket_signal("RELIANCE"), qty=1)
    assert fake.placed_orders[0]["product"] == "CNC"
