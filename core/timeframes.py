import pandas as pd

TIMEFRAME_RULES: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W",
}


def timeframe_to_pandas_rule(timeframe: str) -> str:
    """Map a canonical timeframe string to a pandas resample rule."""
    rule = TIMEFRAME_RULES.get(timeframe)
    if rule is None:
        msg = f"unsupported timeframe: {timeframe}"
        raise ValueError(msg)
    return rule


def resample_bars(
    bars: pd.DataFrame,
    timeframe: str,
    *,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Resample OHLCV bars to a higher timeframe.

    Expects columns: timestamp, open, high, low, close, volume.
    """
    required = {timestamp_col, "open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        msg = f"bars missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    rule = timeframe_to_pandas_rule(timeframe)
    frame = bars.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
    frame = frame.set_index(timestamp_col).sort_index()

    aggregated = frame.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"])
    aggregated = aggregated.reset_index()
    aggregated = aggregated.rename(columns={aggregated.columns[0]: timestamp_col})
    return aggregated
