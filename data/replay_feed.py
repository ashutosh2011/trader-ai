"""Replay bar feed from CSV or DataFrame."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from core.bar import Bar
from data.feed import BarFeed


class ReplayFeed(BarFeed):
    """Replay historical bars from CSV or an in-memory DataFrame."""

    def __init__(self, source: pd.DataFrame | Path | str) -> None:
        if isinstance(source, pd.DataFrame):
            self._frame = _normalize(source)
        else:
            self._frame = _normalize(pd.read_csv(source))

    def bars(self) -> Iterator[Bar]:
        for _, row in self._frame.iterrows():
            yield Bar(
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy()


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"bars missing columns: {sorted(missing)}"
        raise ValueError(msg)
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert("Asia/Kolkata")
    return out.reset_index(drop=True)
