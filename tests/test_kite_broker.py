from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from config.settings import AppSettings
from execution.kite import KiteBroker
from tests.test_risk_manager import _signal

IST = ZoneInfo("Asia/Kolkata")


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
