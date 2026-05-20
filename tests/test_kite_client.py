from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from config.settings import KiteConfig
from data.kite_client import KiteClient, RateLimiter

IST = ZoneInfo("Asia/Kolkata")


def test_kite_client_requires_credentials() -> None:
    with pytest.raises(ValueError, match="api_key"):
        KiteClient(KiteConfig())


def test_kite_client_historical_data() -> None:
    mock_kite = MagicMock()
    mock_kite.historical_data.return_value = [
        {
            "date": datetime(2024, 1, 1, 9, 15, tzinfo=IST),
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100.5,
            "volume": 1000,
        }
    ]
    client = KiteClient(
        KiteConfig(api_key="key", access_token="token"),
        kite=mock_kite,
        rate_limiter=RateLimiter(min_interval_sec=0),
    )
    rows = client.historical_data(
        123,
        datetime(2024, 1, 1, tzinfo=IST),
        datetime(2024, 1, 2, tzinfo=IST),
        "5minute",
    )
    assert len(rows) == 1
    mock_kite.historical_data.assert_called_once()


def test_rate_limiter_waits() -> None:
    limiter = RateLimiter(min_interval_sec=0.01)
    limiter.wait()
    limiter.wait()
