import pandas as pd

from indicators.builtin.sma import SMA
from tests.fixtures.tv_reference import load_fixture, reference_sma
from tests.test_helpers import assert_close, assert_warmup_nan


def test_sma_matches_reference_fixture(synthetic_bars_200: pd.DataFrame) -> None:
    fixture = load_fixture("sma")
    period = int(fixture["params"]["period"])
    computed = SMA(period=period).compute(synthetic_bars_200)
    expected = reference_sma(synthetic_bars_200["close"], period)
    for idx in fixture["indices"]:
        assert_close(float(computed.iloc[idx]), float(expected.iloc[idx]))
        assert_close(float(computed.iloc[idx]), fixture["values"][str(idx)])


def test_sma_warmup(synthetic_bars_200: pd.DataFrame) -> None:
    sma = SMA(period=20)
    result = sma.compute(synthetic_bars_200)
    assert sma.warmup() == 19
    assert_warmup_nan(result, sma.warmup())
