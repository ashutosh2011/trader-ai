"""Structured JSONL trading journal."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from analyst.verdict import Verdict
from core.signal import Signal
from execution.broker import OrderResult
from risk.manager import RiskDecision

logger = structlog.get_logger(__name__)


class TradingJournal:
    """Append-only JSONL journal for signals, risk, verdicts, and orders."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Write a structured event to the journal file and structlog."""
        record = {
            "event": event_type,
            "ts": datetime.now().astimezone().isoformat(),
            **payload,
        }
        logger.info(event_type, **payload)
        if self._path is None:
            return
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def write_signal(self, signal: Signal) -> None:
        self.write_event("signal", {"signal": signal.model_dump(mode="json")})

    def write_risk_decision(self, decision: RiskDecision, symbol: str) -> None:
        self.write_event(
            "risk_decision",
            {"symbol": symbol, "approved": decision.approved, "reason": decision.reason},
        )

    def write_verdict(self, verdict: Verdict, symbol: str) -> None:
        self.write_event(
            "verdict",
            {"symbol": symbol, "verdict": verdict.model_dump(mode="json")},
        )

    def write_order(self, order: OrderResult) -> None:
        self.write_event("order", {"order": order.model_dump(mode="json")})
