"""Prompt builder for the post-trade strategy tuner."""

from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from strategies.registry import list_strategies
from tuner.performance import StrategySymbolPerformance


class TuningContext(BaseModel):
    """Market / operator context passed into the tuning prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: datetime
    notes: str = ""
    lookback_days: int = Field(default=30, ge=1, le=365)


def build_tuning_prompt(
    performances: list[StrategySymbolPerformance],
    ctx: TuningContext,
) -> str:
    """Build the LLM prompt for one tuning review."""
    registry = list_strategies()
    perf_payload = [p.to_prompt_dict() for p in performances]
    schema_hint = {
        "name": "string — short title for this review",
        "summary_rationale": "string — overall read on what worked / failed",
        "recommendations": [
            {
                "symbol": "NSE symbol",
                "current_strategy_id": "registered strategy id",
                "action": "keep | modify_params | switch_strategy | disable",
                "recommended_strategy_id": "required for switch_strategy else null",
                "params": {"example_param": 14},
                "rationale": "why this change given the trade stats",
                "confidence": "0.0-1.0",
            }
        ],
    }
    return f"""You are a quantitative trading strategist reviewing recent trade results.

Your job: propose **structured tuning** per symbol. You may:
- **keep** — strategy and params are fine
- **modify_params** — same strategy, adjust numeric params (e.g. RSI thresholds)
- **switch_strategy** — replace with another registered strategy + params
- **disable** — stop trading this symbol until manually re-enabled

Rules:
1. Return **only** one JSON object matching the schema below. No markdown fences.
2. Use **only** strategy ids from the registry: {json.dumps(registry)}
3. For **modify_params**, include only param keys that exist on that strategy's constructor.
4. Do not invent symbols — only symbols present in PERFORMANCE DATA.
5. Base decisions on win rate, PnL, profit factor, consecutive losses, and recent trades.
6. Prefer conservative changes; one clear action per symbol.
7. If data is thin (<3 trades), prefer **keep** with low confidence.

PERFORMANCE DATA (JSON):
{json.dumps(perf_payload, indent=2)}

CONTEXT:
- as_of: {ctx.as_of.isoformat()}
- lookback_days: {ctx.lookback_days}
- operator_notes: {ctx.notes or "(none)"}

OUTPUT SCHEMA (example shape):
{json.dumps(schema_hint, indent=2)}

Respond with the JSON object only.
"""
