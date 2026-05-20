import pandas as pd
import pytest

from tests.fixtures.bars import make_synthetic_bars


@pytest.fixture
def synthetic_bars_200() -> pd.DataFrame:
    return make_synthetic_bars(200, seed=42)


@pytest.fixture
def synthetic_bars_1000() -> pd.DataFrame:
    return make_synthetic_bars(1000, seed=42)
