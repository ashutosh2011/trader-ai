"""Versioned analyst prompt templates."""

from core.context import Context
from core.signal import Signal

PROMPT_VERSION = "v1"


def build_analyst_prompt(signal: Signal, ctx: Context) -> str:
    """Build the analyst prompt from signal and context snapshot.

    Args:
        signal: Strategy signal under review.
        ctx: Bar context (indicators must not use future bars).

    Returns:
        Prompt string for the LLM provider.
    """
    bar = ctx.current_bar
    snapshot = ", ".join(f"{k}={v:.4f}" for k, v in signal.indicator_snapshot.items())
    reasons = "; ".join(signal.reasons)
    return f"""You are a trading risk analyst (prompt {PROMPT_VERSION}).
Review the signal and respond with JSON only:
{{"action": "APPROVE"|"VETO"|"SHRINK",
  "size_multiplier": 0.0-1.0,
  "confidence": 0.0-1.0,
  "rationale": "..."}}

Rules:
- You may only adjust action, size_multiplier, confidence, rationale.
- Do NOT change entry, stop_loss, or target.
- size_multiplier must be <= 1.0 (never increase size).
- VETO rejects the trade; SHRINK reduces size; APPROVE allows full or reduced size.

Signal:
  symbol: {signal.symbol}
  side: {signal.side}
  entry: {signal.entry:.4f}
  stop_loss: {signal.stop_loss:.4f}
  target: {signal.target:.4f}
  strategy: {signal.strategy_id}
  timeframe: {signal.timeframe}
  confidence: {signal.confidence:.2f}
  reasons: {reasons}
  indicators: {snapshot}
  ts: {signal.ts.isoformat()}

Bar OHLCV:
  open={float(bar['open']):.4f} high={float(bar['high']):.4f}
  low={float(bar['low']):.4f} close={float(bar['close']):.4f}
  volume={float(bar['volume']):.0f}
"""
