import pandas as pd

from indicators.builtin.bbands import BBands
from tests.fixtures.tv_reference import load_fixture, reference_bbands
from tests.test_helpers import assert_close, assert_warmup_nan


def test_bbands_matches_reference_fixture(synthetic_bars_200: pd.DataFrame) -> None:
    fixture = load_fixture("bbands")
    period = int(fixture["params"]["period"])
    mult = float(fixture["params"]["mult"])
    computed = BBands(period=period, mult=mult).compute(synthetic_bars_200)
    expected = reference_bbands(synthetic_bars_200["close"], period, mult)
    for idx in fixture["indices"]:
        for col in ("upper", "middle", "lower"):
            assert_close(float(computed[col].iloc[idx]), float(expected[col].iloc[idx]))
            assert_close(float(computed[col].iloc[idx]), fixture["values"][col][str(idx)])


def test_bbands_warmup(synthetic_bars_200: pd.DataFrame) -> None:
    bb = BBands(period=20)
    result = bb.compute(synthetic_bars_200)
    assert bb.warmup() == 19
    assert_warmup_nan(result["middle"], bb.warmup())
