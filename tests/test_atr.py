import pandas as pd

from indicators.builtin.atr import ATR
from tests.fixtures.tv_reference import load_fixture, reference_atr
from tests.test_helpers import assert_close


def test_atr_matches_reference_fixture(synthetic_bars_200: pd.DataFrame) -> None:
    fixture = load_fixture("atr")
    period = int(fixture["params"]["period"])
    computed = ATR(period=period).compute(synthetic_bars_200)
    expected = reference_atr(synthetic_bars_200, period)
    for idx in fixture["indices"]:
        assert_close(float(computed.iloc[idx]), float(expected.iloc[idx]))
        assert_close(float(computed.iloc[idx]), fixture["values"][str(idx)])


def test_atr_warmup() -> None:
    assert ATR(period=14).warmup() == 14
