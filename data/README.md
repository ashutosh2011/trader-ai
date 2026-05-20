# Data

Bar feeds (replay/live), Kite client, DuckDB candle store, and historical sync.

```python
from data.store import CandleStore
from data.replay_feed import ReplayFeed

store = CandleStore("data/candles.duckdb")
store.upsert_bars("RELIANCE", "5minute", replay_df)
bars = store.get_bars("RELIANCE", "5minute")
```

Live websocket requires `KITE_API_KEY` and `KITE_ACCESS_TOKEN`; without credentials use `LiveKiteFeed(replay_source=df)`.
