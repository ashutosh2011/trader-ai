import pandas as pd

from indicators.builtin.vwap import VWAP
from tests.fixtures.tv_reference import load_fixture, reference_vwap
from tests.test_helpers import assert_close


def test_vwap_matches_reference_fixture(synthetic_bars_200: pd.DataFrame) -> None:
    fixture = load_fixture("vwap")
    computed = VWAP().compute(synthetic_bars_200)
    expected = reference_vwap(synthetic_bars_200)
    for idx in fixture["indices"]:
        assert_close(float(computed.iloc[idx]), float(expected.iloc[idx]))
        assert_close(float(computed.iloc[idx]), fixture["values"][str(idx)])


def test_vwap_warmup() -> None:
    assert VWAP().warmup() == 0
