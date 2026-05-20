import pandas as pd
import pytest

from core.timeframes import resample_bars, timeframe_to_pandas_rule
from tests.fixtures.bars import make_synthetic_bars


def test_timeframe_to_pandas_rule() -> None:
    assert timeframe_to_pandas_rule("5m") == "5min"


def test_timeframe_unsupported() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        timeframe_to_pandas_rule("2m")


def test_resample_bars(synthetic_bars_200: pd.DataFrame) -> None:
    resampled = resample_bars(synthetic_bars_200, "5m")
    assert len(resampled) < len(synthetic_bars_200)
    assert "open" in resampled.columns
