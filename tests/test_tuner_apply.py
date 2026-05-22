"""Tests for applying tuning recommendations."""

from __future__ import annotations

from pathlib import Path

import duckdb

from tuner.active import ActiveConfigStore
from tuner.apply import apply_recommendation
from tuner.schema import StoredRecommendation


def test_apply_modify_params(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "d.duckdb"))
    active = ActiveConfigStore(conn)
    stored = StoredRecommendation(
        id="rec1",
        run_id="run1",
        symbol="INFY",
        current_strategy_id="rsi_mean_revert",
        action="modify_params",
        recommended_strategy_id=None,
        params={"oversold": 28.0},
        rationale="tighter",
        confidence=0.7,
        status="pending",
        created_at="2024-01-01T00:00:00+05:30",
        resolved_at=None,
    )
    cfg = apply_recommendation(active, stored, current_params={"oversold": 30})
    assert cfg.strategy_id == "rsi_mean_revert"
    assert cfg.params["oversold"] == 28.0
    assert cfg.enabled is True
    conn.close()


def test_apply_disable(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "d.duckdb"))
    active = ActiveConfigStore(conn)
    stored = StoredRecommendation(
        id="rec2",
        run_id="run1",
        symbol="INFY",
        current_strategy_id="ema_crossover",
        action="disable",
        recommended_strategy_id=None,
        params={},
        rationale="stop",
        confidence=0.9,
        status="pending",
        created_at="2024-01-01T00:00:00+05:30",
        resolved_at=None,
    )
    cfg = apply_recommendation(active, stored)
    assert cfg.enabled is False
    conn.close()
