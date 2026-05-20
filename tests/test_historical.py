from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd

from data.historical import HistoricalFetcher, detect_gaps, kite_candles_to_dataframe
from data.store import CandleStore

IST = ZoneInfo("Asia/Kolkata")


def test_kite_candles_to_dataframe() -> None:
    rows = [
        {
            "date": datetime(2024, 1, 1, 9, 15, tzinfo=IST),
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": 10,
        }
    ]
    frame = kite_candles_to_dataframe(rows)
    assert len(frame) == 1
    assert "timestamp" in frame.columns


def test_detect_gaps() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 9, 15, tzinfo=IST),
                datetime(2024, 1, 1, 9, 30, tzinfo=IST),
            ],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [1, 1],
        }
    )
    gaps = detect_gaps(frame, "5minute")
    assert len(gaps) == 1


def test_historical_fetch_and_store(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "candles.duckdb"
    store = CandleStore(path)
    mock_client = MagicMock()
    mock_client.historical_data.return_value = [
        {
            "date": datetime(2024, 1, 1, 9, 15, tzinfo=IST),
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
            "volume": 1000,
        }
    ]
    fetcher = HistoricalFetcher(mock_client, store)
    result = fetcher.fetch_and_store(
        symbol="SYNTH",
        instrument_token=1,
        timeframe="5minute",
        from_date=datetime(2024, 1, 1, tzinfo=IST),
        to_date=datetime(2024, 1, 2, tzinfo=IST),
        fill_gaps=False,
    )
    assert result.rows_stored == 1
    store.close()
