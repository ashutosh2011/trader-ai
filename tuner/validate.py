"""Validate and sanitise tuning plans against the strategy registry."""

from __future__ import annotations

import inspect
from typing import Any

from strategies.registry import get_strategy
from tuner.schema import SymbolTuningRecommendation, TuningPlan


def filter_strategy_params(strategy_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by ``strategy_id``'s ``__init__`` (excl. symbol)."""
    cls = get_strategy(strategy_id)
    sig = inspect.signature(cls.__init__)
    allowed = {name for name in sig.parameters if name not in {"self", "symbol"}}
    return {k: v for k, v in params.items() if k in allowed}


def sanitise_plan(plan: TuningPlan) -> TuningPlan:
    """Drop unknown param keys and normalise recommendations."""
    cleaned: list[SymbolTuningRecommendation] = []
    for rec in plan.recommendations:
        target = (
            rec.recommended_strategy_id
            if rec.action == "switch_strategy" and rec.recommended_strategy_id
            else rec.current_strategy_id
        )
        params = (
            filter_strategy_params(target, dict(rec.params))
            if rec.action in {"modify_params", "switch_strategy"}
            else {}
        )
        cleaned.append(
            SymbolTuningRecommendation(
                symbol=rec.symbol,
                current_strategy_id=rec.current_strategy_id,
                action=rec.action,
                recommended_strategy_id=rec.recommended_strategy_id,
                params=params,
                rationale=rec.rationale,
                confidence=rec.confidence,
            )
        )
    return TuningPlan(
        name=plan.name,
        summary_rationale=plan.summary_rationale,
        recommendations=cleaned,
    )
