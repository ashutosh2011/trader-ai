"""Generate JSON reference fixtures from canonical formulas (run once to refresh)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.fixtures.bars import make_synthetic_bars
from tests.fixtures.tv_reference import (
    reference_atr,
    reference_bbands,
    reference_ema,
    reference_macd,
    reference_rsi,
    reference_sma,
    reference_supertrend,
    reference_vwap,
)

FIXTURE_DIR = Path(__file__).resolve().parent
BARS = make_synthetic_bars(200, seed=42)
INDICES = [50, 99, 150, 199]


def _series_values(series: pd.Series, indices: list[int]) -> dict[str, float]:
    return {str(i): float(series.iloc[i]) for i in indices}


def _frame_values(frame: pd.DataFrame, indices: list[int]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        out[col] = {str(i): float(frame[col].iloc[i]) for i in indices}
    return out


def main() -> None:
    close = BARS["close"]
    specs: list[tuple[str, dict[str, object]]] = [
        (
            "ema",
            {"params": {"span": 20}, "values": _series_values(reference_ema(close, 20), INDICES)},
        ),
        (
            "sma",
            {"params": {"period": 20}, "values": _series_values(reference_sma(close, 20), INDICES)},
        ),
        (
            "atr",
            {"params": {"period": 14}, "values": _series_values(reference_atr(BARS, 14), INDICES)},
        ),
        (
            "rsi",
            {
                "params": {"period": 14},
                "values": _series_values(reference_rsi(close, 14), INDICES),
            },
        ),
        (
            "macd",
            {
                "params": {"fast": 12, "slow": 26, "signal": 9},
                "values": _frame_values(reference_macd(close), INDICES),
            },
        ),
        ("vwap", {"params": {}, "values": _series_values(reference_vwap(BARS), INDICES)}),
        (
            "bbands",
            {
                "params": {"period": 20, "mult": 2.0},
                "values": _frame_values(reference_bbands(close, 20, 2.0), INDICES),
            },
        ),
        (
            "supertrend",
            {
                "params": {"period": 10, "multiplier": 3.0},
                "values": _frame_values(reference_supertrend(BARS, 10, 3.0), INDICES),
            },
        ),
    ]
    for name, payload in specs:
        doc = {
            "source": "tests/fixtures/tv_reference.py canonical TradingView-compatible formulas",
            "seed": 42,
            "bar_count": 200,
            "indices": INDICES,
            **payload,
        }
        path = FIXTURE_DIR / f"{name}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
