import pandas as pd

from indicators.builtin.ema import EMA
from tests.fixtures.tv_reference import load_fixture, reference_ema
from tests.test_helpers import assert_close


def test_ema_matches_pandas_at_bar_50(synthetic_bars_200: pd.DataFrame) -> None:
    ema = EMA(span=20)
    computed = ema.compute(synthetic_bars_200)
    expected = synthetic_bars_200["close"].ewm(span=20, adjust=False).mean()
    assert abs(float(computed.iloc[50]) - float(expected.iloc[50])) < 1e-9


def test_ema_matches_reference_fixture(synthetic_bars_200: pd.DataFrame) -> None:
    fixture = load_fixture("ema")
    span = int(fixture["params"]["span"])
    computed = EMA(span=span).compute(synthetic_bars_200)
    expected = reference_ema(synthetic_bars_200["close"], span)
    for idx in fixture["indices"]:
        assert_close(float(computed.iloc[idx]), float(expected.iloc[idx]))
        assert_close(float(computed.iloc[idx]), fixture["values"][str(idx)])


def test_ema_warmup() -> None:
    ema = EMA(span=20)
    assert ema.warmup() == 19
