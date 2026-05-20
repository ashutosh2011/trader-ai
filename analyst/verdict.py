"""Analyst verdict model."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VerdictAction = Literal["APPROVE", "VETO", "SHRINK"]


class Verdict(BaseModel):
    """LLM advisory verdict on a trading signal."""

    model_config = ConfigDict(frozen=True)

    action: VerdictAction
    size_multiplier: float = Field(ge=0, le=1, default=1.0)
    confidence: float = Field(ge=0, le=1, default=0.5)
    rationale: str = ""
    provider: str = "unknown"
    latency_ms: int = Field(ge=0, default=0)
