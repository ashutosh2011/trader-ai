"""Tests for tuner Pydantic schema."""

import pytest

from tuner.schema import SymbolTuningRecommendation, TuningPlan


def test_valid_modify_params() -> None:
    rec = SymbolTuningRecommendation(
        symbol="INFY",
        current_strategy_id="rsi_mean_revert",
        action="modify_params",
        params={"oversold": 25.0},
        rationale="Widen oversold after losses",
        confidence=0.7,
    )
    assert rec.action == "modify_params"


def test_switch_requires_target() -> None:
    with pytest.raises(ValueError, match="recommended_strategy_id"):
        SymbolTuningRecommendation(
            symbol="INFY",
            current_strategy_id="ema_crossover",
            action="switch_strategy",
            rationale="switch",
            confidence=0.5,
        )


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="not registered"):
        SymbolTuningRecommendation(
            symbol="INFY",
            current_strategy_id="ema_crossover",
            action="switch_strategy",
            recommended_strategy_id="not_a_strategy",
            rationale="bad",
            confidence=0.5,
        )


def test_modify_params_empty_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty params"):
        SymbolTuningRecommendation(
            symbol="INFY",
            current_strategy_id="rsi_mean_revert",
            action="modify_params",
            params={},
            rationale="empty",
            confidence=0.5,
        )


def test_tuning_plan_roundtrip() -> None:
    plan = TuningPlan(
        name="Review",
        summary_rationale="ok",
        recommendations=[
            SymbolTuningRecommendation(
                symbol="INFY",
                current_strategy_id="ema_crossover",
                action="keep",
                rationale="fine",
                confidence=0.6,
            )
        ],
    )
    restored = TuningPlan.model_validate_json(plan.model_dump_json())
    assert restored.name == "Review"
