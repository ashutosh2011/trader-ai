"""Synthetic bar generation for demos and CLI."""

import numpy as np
import pandas as pd


def make_synthetic_bars(count: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLCV bars with regime shifts to encourage EMA crosses."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start="2024-01-01 09:15:00",
        periods=count,
        freq="1min",
        tz="Asia/Kolkata",
    )

    returns = rng.normal(0.0, 0.0015, count)
    cycle = np.sin(np.linspace(0.0, 24.0 * np.pi, count))
    returns += 0.0025 * cycle
    for pivot in (count // 4, count // 2, (3 * count) // 4):
        returns[pivot : pivot + 20] += 0.004

    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.empty(count)
    high = np.empty(count)
    low = np.empty(count)
    open_[0] = close[0]
    for i in range(1, count):
        open_[i] = close[i - 1]
    spread = rng.uniform(0.0005, 0.002, count) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000, 10_000, count).astype(float)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "SYNTH",
        }
    )
