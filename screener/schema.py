"""Pydantic models for screener formulas and evaluation results.

The LLM emits a :class:`ScreenerFormula` as structured JSON. All filters
are AND-combined; there is no OR/NOT in v1. The schema is intentionally
narrow so a malformed LLM response fails fast and the runner falls back
to :data:`screener.llm_screener.DEFAULT_FORMULA`.

TRADEOFF: ``compare_to`` only supports a single level of indicator-vs-
indicator comparison (no recursion). That covers the common patterns
("close > SMA(50)", "MACD line > MACD signal") and keeps the prompt
schema tractable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ScreenerTimeframe = Literal["day", "5m"]
ScreenerSideBias = Literal["long", "short", "both"]
ComparisonOp = Literal[">", "<", ">=", "<=", "=="]


class CompareTo(BaseModel):
    """Right-hand side of an :class:`IndicatorFilter` referencing another indicator.

    Only one level of nesting is allowed: a ``CompareTo`` cannot itself
    contain another ``CompareTo``. The whitelist of valid indicator names
    is enforced by the evaluator at compute time, not by the schema, so
    the LLM is encouraged to pick names from the prompt's indicator list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicator: str
    params: dict[str, float] = Field(default_factory=dict)

    @field_validator("indicator")
    @classmethod
    def indicator_non_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "compare_to.indicator must be non-empty"
            raise ValueError(msg)
        return value


class IndicatorFilter(BaseModel):
    """Filter that compares an indicator's latest value to a constant or another indicator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["indicator"] = "indicator"
    indicator: str
    params: dict[str, float] = Field(default_factory=dict)
    op: ComparisonOp
    value: float | None = None
    compare_to: CompareTo | None = None

    @field_validator("indicator")
    @classmethod
    def indicator_non_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "indicator must be non-empty"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def exactly_one_rhs(self) -> IndicatorFilter:
        has_value = self.value is not None
        has_compare = self.compare_to is not None
        if has_value and has_compare:
            msg = "IndicatorFilter requires exactly one of 'value' or 'compare_to' (got both)"
            raise ValueError(msg)
        if not has_value and not has_compare:
            msg = "IndicatorFilter requires exactly one of 'value' or 'compare_to' (got neither)"
            raise ValueError(msg)
        return self


class VolumeFilter(BaseModel):
    """Filter on the most recent volume bar.

    Either an absolute threshold (``value``) or a multiple of the rolling
    mean over ``avg_window`` (``value_x_avg``) must be specified. The two
    forms are mutually exclusive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["volume"] = "volume"
    op: ComparisonOp
    value: float | None = None
    value_x_avg: float | None = None
    avg_window: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def exactly_one_form(self) -> VolumeFilter:
        has_value = self.value is not None
        has_x_avg = self.value_x_avg is not None
        if has_value and has_x_avg:
            msg = "VolumeFilter requires exactly one of 'value' or 'value_x_avg' (got both)"
            raise ValueError(msg)
        if not has_value and not has_x_avg:
            msg = "VolumeFilter requires exactly one of 'value' or 'value_x_avg' (got neither)"
            raise ValueError(msg)
        return self


class PriceChangeFilter(BaseModel):
    """Filter on percent change over a bar lookback window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["price_change"] = "price_change"
    window: int = Field(ge=1)
    op: ComparisonOp
    value_pct: float


ScreenerFilter = Annotated[
    IndicatorFilter | VolumeFilter | PriceChangeFilter,
    Field(discriminator="type"),
]


class ScreenerFormula(BaseModel):
    """Structured filter formula returned by the LLM screener.

    All filters are AND-combined; a symbol passes only when every filter
    evaluates true on the last closed bar. ``rationale`` is displayed
    verbatim on the dashboard.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    timeframe: ScreenerTimeframe
    side_bias: ScreenerSideBias
    rationale: str
    filters: list[ScreenerFilter]

    @field_validator("filters")
    @classmethod
    def filters_non_empty(cls, value: list[ScreenerFilter]) -> list[ScreenerFilter]:
        if not value:
            msg = "filters must contain at least one entry"
            raise ValueError(msg)
        return value

    @field_validator("name")
    @classmethod
    def name_non_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "name must be non-empty"
            raise ValueError(msg)
        return value


class ScreeningMatch(BaseModel):
    """One filter's evaluation result for a passing symbol.

    ``threshold`` is a string for the volume-x-avg case (e.g. ``"1.5×avg(20)"``)
    and a float for plain numeric thresholds. ``passed`` is always True in
    persisted results but kept for debug / future enrichment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filter_index: int = Field(ge=0)
    value: float
    threshold: float | str
    passed: bool


class ScreeningResult(BaseModel):
    """One symbol's full result row from an evaluator pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side_bias: ScreenerSideBias
    matches: list[ScreeningMatch]
    bars_evaluated: int = Field(ge=0)
    last_bar_ts: datetime


__all__ = [
    "ComparisonOp",
    "CompareTo",
    "IndicatorFilter",
    "PriceChangeFilter",
    "ScreenerFilter",
    "ScreenerFormula",
    "ScreenerSideBias",
    "ScreenerTimeframe",
    "ScreeningMatch",
    "ScreeningResult",
    "VolumeFilter",
]
