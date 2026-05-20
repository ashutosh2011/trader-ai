import pandas as pd

from indicators.builtin.macd import MACD
from tests.fixtures.tv_reference import load_fixture, reference_macd
from tests.test_helpers import assert_close


def test_macd_matches_reference_fixture(synthetic_bars_200: pd.DataFrame) -> None:
    fixture = load_fixture("macd")
    fast = int(fixture["params"]["fast"])
    slow = int(fixture["params"]["slow"])
    signal = int(fixture["params"]["signal"])
    computed = MACD(fast=fast, slow=slow, signal=signal).compute(synthetic_bars_200)
    expected = reference_macd(synthetic_bars_200["close"], fast, slow, signal)
    for idx in fixture["indices"]:
        for col in ("macd", "signal", "histogram"):
            assert_close(float(computed[col].iloc[idx]), float(expected[col].iloc[idx]))
            assert_close(float(computed[col].iloc[idx]), fixture["values"][col][str(idx)])


def test_macd_warmup() -> None:
    assert MACD().warmup() == 33
