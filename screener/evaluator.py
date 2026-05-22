"""Pure evaluation of a :class:`ScreenerFormula` against per-symbol candles.

This module has no I/O, no network, no INFO-level logging. It receives
already-loaded OHLCV DataFrames and returns a list of
:class:`ScreeningResult` for symbols where **every** filter passes.

Symbols are silently skipped (with a debug log) when:
    * fewer than ``min_bars`` rows are present;
    * required OHLCV columns are missing;
    * an indicator raises during compute (e.g. NaN tail);
    * a volume rolling mean is 0 / NaN when the filter requested
      ``value_x_avg``.

That last rule mirrors the spec — partial data shouldn't crash the run,
just shrink the candidate set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import cast
from zoneinfo import ZoneInfo

import pandas as pd
import structlog

from screener.registry import resolve_indicator_value
from screener.schema import (
    ComparisonOp,
    IndicatorFilter,
    PriceChangeFilter,
    ScreenerFilter,
    ScreenerFormula,
    ScreeningMatch,
    ScreeningResult,
    VolumeFilter,
)

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

REQUIRED_OHLCV: frozenset[str] = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"},
)


def evaluate(
    formula: ScreenerFormula,
    candles_by_symbol: Mapping[str, pd.DataFrame],
    *,
    min_bars: int = 60,
) -> list[ScreeningResult]:
    """Evaluate ``formula`` against each symbol's candles.

    Args:
        formula: The screener formula to apply.
        candles_by_symbol: Mapping from symbol to an OHLCV DataFrame
            (columns ``timestamp``, ``open``, ``high``, ``low``, ``close``,
            ``volume``; ``timestamp`` must be tz-aware).
        min_bars: Minimum bars required per symbol before evaluation.

    Returns:
        Symbols where all filters pass, each represented as a
        :class:`ScreeningResult` whose ``matches`` list has the same
        length as ``formula.filters``.
    """
    results: list[ScreeningResult] = []
    for symbol in sorted(candles_by_symbol):
        candles = candles_by_symbol[symbol]
        if not _symbol_is_evaluable(symbol, candles, min_bars=min_bars):
            continue
        outcome = _evaluate_symbol(formula, symbol, candles)
        if outcome is not None:
            results.append(outcome)
    return results


def _symbol_is_evaluable(
    symbol: str,
    candles: pd.DataFrame,
    *,
    min_bars: int,
) -> bool:
    if len(candles) < min_bars:
        logger.debug(
            "screener_symbol_skipped_too_few_bars",
            symbol=symbol,
            bars=len(candles),
            min_bars=min_bars,
        )
        return False
    missing = REQUIRED_OHLCV - set(candles.columns)
    if missing:
        logger.debug(
            "screener_symbol_skipped_missing_columns",
            symbol=symbol,
            missing=sorted(missing),
        )
        return False
    return True


def _evaluate_symbol(
    formula: ScreenerFormula,
    symbol: str,
    candles: pd.DataFrame,
) -> ScreeningResult | None:
    matches: list[ScreeningMatch] = []
    for idx, screener_filter in enumerate(formula.filters):
        try:
            match = _evaluate_filter(screener_filter, candles, filter_index=idx)
        except ValueError as exc:
            logger.debug(
                "screener_filter_error",
                symbol=symbol,
                filter_index=idx,
                error=str(exc),
            )
            return None
        if match is None:
            return None
        matches.append(match)

    last_ts = _last_timestamp(candles)
    return ScreeningResult(
        symbol=symbol,
        side_bias=formula.side_bias,
        matches=matches,
        bars_evaluated=int(len(candles)),
        last_bar_ts=last_ts,
    )


def _evaluate_filter(
    screener_filter: ScreenerFilter,
    candles: pd.DataFrame,
    *,
    filter_index: int,
) -> ScreeningMatch | None:
    if isinstance(screener_filter, IndicatorFilter):
        return _evaluate_indicator_filter(screener_filter, candles, filter_index=filter_index)
    if isinstance(screener_filter, VolumeFilter):
        return _evaluate_volume_filter(screener_filter, candles, filter_index=filter_index)
    if isinstance(screener_filter, PriceChangeFilter):
        return _evaluate_price_change_filter(
            screener_filter,
            candles,
            filter_index=filter_index,
        )
    # Exhaustive guard for future filter variants.
    msg = f"unknown filter type: {type(screener_filter).__name__}"
    raise ValueError(msg)


def _evaluate_indicator_filter(
    screener_filter: IndicatorFilter,
    candles: pd.DataFrame,
    *,
    filter_index: int,
) -> ScreeningMatch | None:
    lhs = resolve_indicator_value(
        screener_filter.indicator,
        screener_filter.params,
        candles,
    )
    if screener_filter.compare_to is not None:
        rhs = resolve_indicator_value(
            screener_filter.compare_to.indicator,
            screener_filter.compare_to.params,
            candles,
        )
        threshold: float | str = rhs
    else:
        if screener_filter.value is None:  # pragma: no cover - schema-enforced
            msg = "IndicatorFilter.value missing despite passing validation"
            raise ValueError(msg)
        rhs = float(screener_filter.value)
        threshold = rhs

    if not _compare(lhs, screener_filter.op, rhs):
        return None
    return ScreeningMatch(
        filter_index=filter_index,
        value=lhs,
        threshold=threshold,
        passed=True,
    )


def _evaluate_volume_filter(
    screener_filter: VolumeFilter,
    candles: pd.DataFrame,
    *,
    filter_index: int,
) -> ScreeningMatch | None:
    volume_series = candles["volume"]
    if len(volume_series) == 0:
        return None
    last_volume = float(cast(float, volume_series.iloc[-1]))
    if math.isnan(last_volume):
        return None

    if screener_filter.value_x_avg is not None:
        window = screener_filter.avg_window
        if len(volume_series) < window:
            return None
        rolling_mean_raw = (
            volume_series.rolling(window=window, min_periods=window).mean().iloc[-1]
        )
        rolling_mean = float(cast(float, rolling_mean_raw))
        if math.isnan(rolling_mean) or rolling_mean == 0.0:
            return None
        rhs = rolling_mean * float(screener_filter.value_x_avg)
        threshold: float | str = f"{screener_filter.value_x_avg}×avg({window})"
    else:
        if screener_filter.value is None:  # pragma: no cover - schema-enforced
            msg = "VolumeFilter.value missing despite passing validation"
            raise ValueError(msg)
        rhs = float(screener_filter.value)
        threshold = rhs

    if not _compare(last_volume, screener_filter.op, rhs):
        return None
    return ScreeningMatch(
        filter_index=filter_index,
        value=last_volume,
        threshold=threshold,
        passed=True,
    )


def _evaluate_price_change_filter(
    screener_filter: PriceChangeFilter,
    candles: pd.DataFrame,
    *,
    filter_index: int,
) -> ScreeningMatch | None:
    close = candles["close"]
    window = screener_filter.window
    if len(close) <= window:
        return None
    current = float(cast(float, close.iloc[-1]))
    prior = float(cast(float, close.iloc[-1 - window]))
    if math.isnan(current) or math.isnan(prior) or prior == 0.0:
        return None
    pct_change = (current - prior) / prior * 100.0
    if not _compare(pct_change, screener_filter.op, screener_filter.value_pct):
        return None
    return ScreeningMatch(
        filter_index=filter_index,
        value=pct_change,
        threshold=float(screener_filter.value_pct),
        passed=True,
    )


def _compare(lhs: float, op: ComparisonOp, rhs: float) -> bool:
    if op == ">":
        return lhs > rhs
    if op == "<":
        return lhs < rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<=":
        return lhs <= rhs
    if op == "==":
        return lhs == rhs
    # Schema is Literal-typed; unreachable but defensive.
    msg = f"unknown comparison op: {op}"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def _last_timestamp(candles: pd.DataFrame) -> datetime:
    """Return the tz-aware (IST) datetime of the last bar."""
    ts_raw = candles["timestamp"].iloc[-1]
    ts = pd.Timestamp(ts_raw)
    ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
    return ts.to_pydatetime()


__all__ = ["evaluate"]
