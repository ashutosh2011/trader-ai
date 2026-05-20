from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from data.store import CandleStore

IST = ZoneInfo("Asia/Kolkata")


def _sample_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 9, 15, tzinfo=IST),
                datetime(2024, 1, 1, 9, 16, tzinfo=IST),
            ],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
        }
    )


def test_candle_store_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "test.duckdb"
    store = CandleStore(db)
    store.upsert_bars("SYNTH", "1m", _sample_bars())
    loaded = store.get_bars("SYNTH", "1m")
    assert len(loaded) == 2
    ts = store.latest_timestamp("SYNTH", "1m")
    assert ts is not None
    keys = store.symbols()
    assert len(keys) == 1
    store.close()
