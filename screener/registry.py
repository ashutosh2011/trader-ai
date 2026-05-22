"""Whitelist of screener-facing indicator names and their resolvers.

The screener intentionally exposes a narrow, named indicator surface to
the LLM. Each whitelist entry maps an LLM-facing name (e.g. ``"rsi"``,
``"macd_hist"``) to either a registered :class:`indicators.base.Indicator`
class plus a column selector, or to a bare OHLCV column (``"close"`` etc.).

``resolve_indicator_value`` returns the last (most recent) non-NaN scalar
value, raising :class:`ValueError` on unknown name, invalid params, or
NaN at the tail.

TRADEOFF: We do not expose every indicator parameter to the LLM. The
spec is "what the LLM picks should compile to a single scalar on the
last closed bar"; multi-column indicators like MACD/BBands/Supertrend
are flattened into one entry per column (``macd_line``, ``macd_signal``,
``macd_hist``, ``bb_upper`` / ``bb_middle`` / ``bb_lower``,
``supertrend_dir``).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict

from indicators.base import Indicator
from indicators.builtin.atr import ATR
from indicators.builtin.bbands import BBands
from indicators.builtin.ema import EMA
from indicators.builtin.macd import MACD
from indicators.builtin.rsi import RSI
from indicators.builtin.sma import SMA
from indicators.builtin.supertrend import Supertrend

IndicatorOutputKind = Literal["series", "frame", "ohlc"]

OHLC_COLUMNS: frozenset[str] = frozenset({"open", "high", "low", "close"})


@dataclass(frozen=True)
class IndicatorSpec:
    """Whitelist entry mapping an LLM name to a resolver.

    Attributes:
        cls: The :class:`Indicator` class to instantiate, or ``None`` for
            raw OHLC columns.
        output_kind: ``"series"`` (Indicator returns a Series),
            ``"frame"`` (Indicator returns a DataFrame; ``series_column``
            picks one column), or ``"ohlc"`` (read OHLCV column directly).
        series_column: Column name to extract when ``output_kind == "frame"``
            or ``output_kind == "ohlc"``.
    """

    cls: type[Indicator] | None
    output_kind: IndicatorOutputKind
    series_column: str | None


class IndicatorParams(BaseModel):
    """Coerced indicator parameters (typed ints/floats)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    int_params: dict[str, int]
    float_params: dict[str, float]


INDICATOR_WHITELIST: dict[str, IndicatorSpec] = {
    "rsi": IndicatorSpec(cls=RSI, output_kind="series", series_column=None),
    "sma": IndicatorSpec(cls=SMA, output_kind="series", series_column=None),
    "ema": IndicatorSpec(cls=EMA, output_kind="series", series_column=None),
    "atr": IndicatorSpec(cls=ATR, output_kind="series", series_column=None),
    "close": IndicatorSpec(cls=None, output_kind="ohlc", series_column="close"),
    "open": IndicatorSpec(cls=None, output_kind="ohlc", series_column="open"),
    "high": IndicatorSpec(cls=None, output_kind="ohlc", series_column="high"),
    "low": IndicatorSpec(cls=None, output_kind="ohlc", series_column="low"),
    "macd_line": IndicatorSpec(cls=MACD, output_kind="frame", series_column="macd"),
    "macd_signal": IndicatorSpec(cls=MACD, output_kind="frame", series_column="signal"),
    "macd_hist": IndicatorSpec(cls=MACD, output_kind="frame", series_column="histogram"),
    "bb_upper": IndicatorSpec(cls=BBands, output_kind="frame", series_column="upper"),
    "bb_middle": IndicatorSpec(cls=BBands, output_kind="frame", series_column="middle"),
    "bb_lower": IndicatorSpec(cls=BBands, output_kind="frame", series_column="lower"),
    "supertrend_dir": IndicatorSpec(
        cls=Supertrend, output_kind="frame", series_column="direction"
    ),
}


def whitelist_names() -> list[str]:
    """Return the sorted list of LLM-facing indicator names."""
    return sorted(INDICATOR_WHITELIST)


def _coerce_params(spec: IndicatorSpec, params: dict[str, float]) -> dict[str, int | float]:
    """Coerce raw float params to the integer kwargs the indicator expects.

    All ``Indicator`` subclasses we whitelist take ``int``-valued periods
    (``period``, ``span``, ``fast``, ``slow``, ``signal``, ``multiplier``)
    except :class:`BBands.mult` and :class:`Supertrend.multiplier` which
    accept floats. We use ``inspect.signature`` to discover the declared
    annotation and route accordingly.
    """
    if spec.cls is None:
        return {}
    sig = inspect.signature(spec.cls.__init__)
    coerced: dict[str, int | float] = {}
    for key, value in params.items():
        if key not in sig.parameters:
            msg = f"unknown parameter {key!r} for {spec.cls.__name__}"
            raise ValueError(msg)
        param = sig.parameters[key]
        annotation = param.annotation
        if annotation is int:
            if not float(value).is_integer():
                msg = f"parameter {key!r} for {spec.cls.__name__} must be an integer"
                raise ValueError(msg)
            coerced[key] = int(value)
        elif annotation is float:
            coerced[key] = float(value)
        else:
            coerced[key] = value
    return coerced


def _instantiate(spec: IndicatorSpec, params: dict[str, float]) -> Indicator:
    if spec.cls is None:
        msg = "spec has no Indicator class (OHLC column)"
        raise ValueError(msg)
    kwargs = _coerce_params(spec, params)
    try:
        return spec.cls(**kwargs)
    except TypeError as exc:
        msg = f"invalid params for {spec.cls.__name__}: {exc}"
        raise ValueError(msg) from exc


def resolve_indicator_value(
    name: str,
    params: dict[str, float],
    candles: pd.DataFrame,
) -> float:
    """Return the last value of the whitelisted indicator on ``candles``.

    Args:
        name: One of :func:`whitelist_names`.
        params: Parameters to forward to the indicator constructor.
            Unknown keys raise :class:`ValueError`.
        candles: OHLCV DataFrame.

    Returns:
        The most recent scalar value as a Python float.

    Raises:
        ValueError: If ``name`` is not whitelisted, the params are invalid,
            the OHLCV column is missing, or the last value is NaN.
    """
    if name not in INDICATOR_WHITELIST:
        msg = (
            f"indicator {name!r} not in screener whitelist; "
            f"valid names: {whitelist_names()}"
        )
        raise ValueError(msg)
    spec = INDICATOR_WHITELIST[name]

    if spec.output_kind == "ohlc":
        column = spec.series_column or name
        if column not in candles.columns:
            msg = f"candles missing required column {column!r}"
            raise ValueError(msg)
        series = candles[column]
        if len(series) == 0:
            msg = f"cannot resolve {name!r} on empty candles"
            raise ValueError(msg)
        last_raw = series.iloc[-1]
    else:
        indicator = _instantiate(spec, params)
        output = indicator.compute(candles)
        if spec.output_kind == "frame":
            if not isinstance(output, pd.DataFrame):
                msg = (
                    f"indicator {name!r}: expected DataFrame output, "
                    f"got {type(output).__name__}"
                )
                raise ValueError(msg)
            column = spec.series_column or name
            if column not in output.columns:
                msg = (
                    f"indicator {name!r}: output frame missing column "
                    f"{column!r}; got {list(output.columns)}"
                )
                raise ValueError(msg)
            series = output[column]
        else:
            if not isinstance(output, pd.Series):
                msg = (
                    f"indicator {name!r}: expected Series output, "
                    f"got {type(output).__name__}"
                )
                raise ValueError(msg)
            series = output
        if len(series) == 0:
            msg = f"indicator {name!r} produced no values"
            raise ValueError(msg)
        last_raw = series.iloc[-1]

    last = float(cast(float, last_raw))
    if math.isnan(last):
        msg = f"indicator {name!r} last value is NaN (insufficient warmup?)"
        raise ValueError(msg)
    return last


__all__ = [
    "INDICATOR_WHITELIST",
    "IndicatorSpec",
    "resolve_indicator_value",
    "whitelist_names",
]
