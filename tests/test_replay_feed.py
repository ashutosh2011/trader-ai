from data.replay_feed import ReplayFeed
from tests.fixtures.bars import make_synthetic_bars


def test_replay_feed_bars_and_frame() -> None:
    frame = make_synthetic_bars(20)
    feed = ReplayFeed(frame)
    bars = list(feed.bars())
    assert len(bars) == 20
    assert feed.to_dataframe().shape[0] == 20
