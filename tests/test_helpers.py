import math

import pandas as pd


def assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual - expected) < tol


def assert_warmup_nan(series: pd.Series, warmup: int) -> None:
    if warmup <= 0:
        return
    assert series.iloc[:warmup].isna().all()
