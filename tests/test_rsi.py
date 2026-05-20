import pandas as pd

from indicators.builtin.rsi import RSI
from tests.fixtures.tv_reference import load_fixture, reference_rsi
from tests.test_helpers import assert_close, assert_warmup_nan


def test_rsi_matches_reference_fixture(synthetic_bars_200: pd.DataFrame) -> None:
    fixture = load_fixture("rsi")
    period = int(fixture["params"]["period"])
    computed = RSI(period=period).compute(synthetic_bars_200)
    expected = reference_rsi(synthetic_bars_200["close"], period)
    for idx in fixture["indices"]:
        assert_close(float(computed.iloc[idx]), float(expected.iloc[idx]))
        assert_close(float(computed.iloc[idx]), fixture["values"][str(idx)])


def test_rsi_warmup(synthetic_bars_200: pd.DataFrame) -> None:
    rsi = RSI(period=14)
    result = rsi.compute(synthetic_bars_200)
    assert rsi.warmup() == 13
    assert_warmup_nan(result, rsi.warmup())
