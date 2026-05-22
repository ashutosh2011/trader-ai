"""Apply a pending tuning recommendation to active symbol config."""

from __future__ import annotations

from typing import Any

import structlog

from tuner.active import ActiveConfigStore, SymbolActiveConfig
from tuner.schema import (
    StoredRecommendation,
    SymbolTuningRecommendation,
    effective_strategy_id,
    merged_params,
)
from tuner.validate import filter_strategy_params

logger = structlog.get_logger(__name__)


def recommendation_to_symbol_rec(stored: StoredRecommendation) -> SymbolTuningRecommendation:
    """Convert a stored row back to a :class:`SymbolTuningRecommendation`."""
    return SymbolTuningRecommendation(
        symbol=stored.symbol,
        current_strategy_id=stored.current_strategy_id,
        action=stored.action,
        recommended_strategy_id=stored.recommended_strategy_id,
        params=stored.params,
        rationale=stored.rationale,
        confidence=stored.confidence,
    )


def apply_recommendation(
    active: ActiveConfigStore,
    stored: StoredRecommendation,
    *,
    current_params: dict[str, Any] | None = None,
) -> SymbolActiveConfig:
    """Apply one recommendation to :class:`ActiveConfigStore`.

    Args:
        active: Target config store.
        stored: Pending recommendation row.
        current_params: Params to merge for ``modify_params`` when no
            active config exists yet.

    Returns:
        The upserted active config (or disabled row for ``disable``).

    Raises:
        ValueError: If action/params are inconsistent after sanitisation.
    """
    rec = recommendation_to_symbol_rec(stored)
    existing = active.get(stored.symbol)
    base_params = current_params or (existing.params if existing else {})

    if rec.action == "keep":
        strategy_id = rec.current_strategy_id
        params = filter_strategy_params(strategy_id, base_params)
        enabled = existing.enabled if existing else True
    elif rec.action == "disable":
        strategy_id = rec.current_strategy_id
        params = filter_strategy_params(strategy_id, base_params)
        enabled = False
    else:
        strategy_id = effective_strategy_id(rec)
        params = filter_strategy_params(
            strategy_id,
            merged_params(rec, current_params=base_params),
        )
        if rec.action == "modify_params" and not params:
            msg = "modify_params produced empty params after filtering"
            raise ValueError(msg)
        enabled = True

    return active.upsert(
        symbol=stored.symbol,
        strategy_id=strategy_id,
        params=params,
        enabled=enabled,
        source_recommendation_id=stored.id,
    )
