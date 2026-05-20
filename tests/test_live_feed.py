from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from data.live_feed import LiveKiteFeed
from tests.fixtures.bars import make_synthetic_bars

IST = ZoneInfo("Asia/Kolkata")


@pytest.mark.asyncio
async def test_live_feed_replay_mode() -> None:
    bars = make_synthetic_bars(10, seed=1)
    feed = LiveKiteFeed(replay_source=bars)
    await feed.connect()
    assert feed.is_connected()
    df = feed.to_dataframe()
    assert len(df) == 10
    await feed.disconnect()
    assert not feed.is_connected()


def test_live_feed_ingest_ticks() -> None:
    feed = LiveKiteFeed()
    ts = int(datetime(2024, 1, 1, 10, 0, tzinfo=IST).timestamp())
    feed.ingest_ticks(
        [
            {
                "timestamp": ts,
                "last_price": 101.0,
                "volume": 500,
                "ohlc": {"open": 100, "high": 102, "low": 99, "close": 101},
            }
        ]
    )
    bars = list(feed.bars())
    assert len(bars) == 1
    assert bars[0].close == 101.0
