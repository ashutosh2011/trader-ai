"""Pydantic models for LLM tuning recommendations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strategies.registry import get_strategy, list_strategies

TuningAction = Literal["keep", "modify_params", "switch_strategy", "disable"]


class SymbolTuningRecommendation(BaseModel):
    """One symbol-level recommendation from the LLM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    current_strategy_id: str = Field(min_length=1)
    action: TuningAction
    recommended_strategy_id: str | None = None
    params: dict[str, int | float | str] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    @model_validator(mode="after")
    def _validate_action_shape(self) -> SymbolTuningRecommendation:
        if self.action == "keep":
            return self
        if self.action == "disable":
            return self
        if self.action == "switch_strategy":
            if not self.recommended_strategy_id:
                msg = "switch_strategy requires recommended_strategy_id"
                raise ValueError(msg)
            _assert_strategy_registered(self.recommended_strategy_id)
            return self
        if self.action == "modify_params":
            if not self.params:
                msg = "modify_params requires non-empty params"
                raise ValueError(msg)
            target = self.recommended_strategy_id or self.current_strategy_id
            _assert_strategy_registered(target)
            return self
        msg = f"unknown action: {self.action}"
        raise ValueError(msg)


class TuningPlan(BaseModel):
    """Full tuning output from one LLM call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    summary_rationale: str = Field(min_length=1)
    recommendations: list[SymbolTuningRecommendation] = Field(default_factory=list)


RecommendationStatus = Literal["pending", "applied", "rejected"]


class StoredRecommendation(BaseModel):
    """One persisted recommendation row (includes bookkeeping)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    run_id: str
    symbol: str
    current_strategy_id: str
    action: TuningAction
    recommended_strategy_id: str | None
    params: dict[str, int | float | str]
    rationale: str
    confidence: float
    status: RecommendationStatus
    created_at: str
    resolved_at: str | None = None


def _assert_strategy_registered(strategy_id: str) -> None:
    registered = list_strategies()
    if strategy_id not in registered:
        msg = (
            f"strategy not registered: {strategy_id!r}; "
            f"allowed: {registered}"
        )
        raise ValueError(msg)
    get_strategy(strategy_id)


def effective_strategy_id(rec: SymbolTuningRecommendation) -> str:
    """Return the strategy id that would be active after applying ``rec``."""
    if rec.action == "switch_strategy" and rec.recommended_strategy_id:
        return rec.recommended_strategy_id
    return rec.current_strategy_id


def merged_params(
    rec: SymbolTuningRecommendation,
    *,
    current_params: dict[str, Any],
) -> dict[str, Any]:
    """Merge ``current_params`` with recommendation params for modify/switch."""
    if rec.action in {"keep", "disable"}:
        return dict(current_params)
    base = dict(current_params)
    base.update(rec.params)
    return base
