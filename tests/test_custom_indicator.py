import pandas as pd

import indicators.custom  # noqa: F401 — register custom
from indicators.custom.example_momentum import PriceMomentum
from indicators.registry import get_indicator, list_indicators


def test_price_momentum_registered() -> None:
    assert "price_momentum" in list_indicators()
    assert get_indicator("price_momentum") is PriceMomentum


def test_price_momentum_compute(synthetic_bars_200: pd.DataFrame) -> None:
    ind = PriceMomentum(period=5)
    series = ind.compute(synthetic_bars_200)
    assert len(series) == len(synthetic_bars_200)
    assert pd.isna(series.iloc[0])
    assert ind.warmup() == 5
    expected = synthetic_bars_200["close"] - synthetic_bars_200["close"].shift(5)
    pd.testing.assert_series_equal(series, expected, check_names=False)
