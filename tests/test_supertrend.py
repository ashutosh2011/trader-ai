import pandas as pd

from indicators.builtin.supertrend import Supertrend
from tests.fixtures.tv_reference import load_fixture, reference_supertrend
from tests.test_helpers import assert_close


def test_supertrend_matches_reference_fixture(synthetic_bars_200: pd.DataFrame) -> None:
    fixture = load_fixture("supertrend")
    period = int(fixture["params"]["period"])
    multiplier = float(fixture["params"]["multiplier"])
    computed = Supertrend(period=period, multiplier=multiplier).compute(synthetic_bars_200)
    expected = reference_supertrend(synthetic_bars_200, period, multiplier)
    for idx in fixture["indices"]:
        for col in ("supertrend", "upper", "lower"):
            assert_close(float(computed[col].iloc[idx]), float(expected[col].iloc[idx]))
            assert_close(float(computed[col].iloc[idx]), fixture["values"][col][str(idx)])
        assert int(computed["direction"].iloc[idx]) == int(expected["direction"].iloc[idx])


def test_supertrend_warmup() -> None:
    assert Supertrend(period=10).warmup() == 10
