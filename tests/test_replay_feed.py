from pathlib import Path
from zoneinfo import ZoneInfo

from data.replay_feed import ReplayFeed
from tests.fixtures.bars import make_synthetic_bars

IST = ZoneInfo("Asia/Kolkata")


def test_replay_feed_bars_and_frame() -> None:
    frame = make_synthetic_bars(20)
    feed = ReplayFeed(frame)
    bars = list(feed.bars())
    assert len(bars) == 20
    assert feed.to_dataframe().shape[0] == 20


def test_replay_feed_csv_naive_timestamps_localized_to_ist(tmp_path: Path) -> None:
    csv = tmp_path / "bars.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 09:15:00,100,101,99,100.5,1000\n"
        "2024-01-01 09:16:00,100.5,101.5,100,101,1500\n",
        encoding="utf-8",
    )

    feed = ReplayFeed(csv)
    df = feed.to_dataframe()

    assert df.shape[0] == 2
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "Asia/Kolkata"
    assert df["timestamp"].iloc[0] == df["timestamp"].iloc[0].tz_convert(IST)
    assert df["timestamp"].iloc[0].hour == 9
    assert df["timestamp"].iloc[0].minute == 15


def test_replay_feed_csv_tzaware_timestamps_converted_to_ist(tmp_path: Path) -> None:
    csv = tmp_path / "bars_tz.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01 03:45:00+00:00,100,101,99,100.5,1000\n"
        "2024-01-01 03:46:00+00:00,100.5,101.5,100,101,1500\n",
        encoding="utf-8",
    )

    feed = ReplayFeed(csv)
    df = feed.to_dataframe()

    assert df.shape[0] == 2
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "Asia/Kolkata"
    assert df["timestamp"].iloc[0].hour == 9
    assert df["timestamp"].iloc[0].minute == 15
    assert df["timestamp"].iloc[1].hour == 9
    assert df["timestamp"].iloc[1].minute == 16
