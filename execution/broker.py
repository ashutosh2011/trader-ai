"""Broker abstract base class and shared order models."""

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.signal import Signal

OrderStatus = Literal["FILLED", "REJECTED", "PENDING"]


class Position(BaseModel):
    """Open position snapshot.

    ``stop_loss`` and ``target`` are optional because broker-reported
    positions (e.g. :class:`KiteBroker.get_positions`) do not carry the
    bracket levels — those live on the child orders / GTT triggers and
    are back-filled by :class:`execution.reconciler.StateReconciler`.

    TRADEOFF: callers must treat ``None`` levels as "unknown" and skip
    risk math that needs them (per-share stop distance, target distance).
    Callers that always synthesise their own bracket (paper broker,
    backtest engine) populate both fields and behave as before.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: Literal["LONG", "SHORT"]
    qty: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss: float | None = None
    target: float | None = None
    strategy_id: str
    opened_at: datetime


class OrderResult(BaseModel):
    """Result of placing a bracket order."""

    model_config = ConfigDict(frozen=True)

    client_order_id: str
    status: OrderStatus
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int
    fill_price: float
    message: str = ""


def deterministic_client_order_id(
    strategy_id: str,
    signal_ts: datetime,
    symbol: str,
) -> str:
    """Idempotent client order ID from strategy, timestamp, and symbol."""
    payload = f"{strategy_id}|{signal_ts.isoformat()}|{symbol}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"tb-{digest}"


class Broker(ABC):
    """Abstract broker interface for live and paper execution."""

    @abstractmethod
    def place_bracket_order(self, signal: Signal, qty: int) -> OrderResult:
        """Place entry with attached stop-loss and target (logical bracket)."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return currently open positions."""

    @abstractmethod
    def flatten_all(self) -> None:
        """Close all open positions at market."""


class FlattenIncomplete(Exception):
    """Raised by :meth:`Broker.flatten_all` when positions remain open.

    ``open_positions`` carries the broker-reported residual positions after
    the configured poll budget is exhausted, so the caller (typically the
    ``flatten`` CLI subcommand or a kill-switch recovery script) can surface
    a precise, actionable error.
    """

    def __init__(self, open_positions: list[Position], attempts: int) -> None:
        self.open_positions = open_positions
        self.attempts = attempts
        symbols = ",".join(p.symbol for p in open_positions) or "<none>"
        super().__init__(
            f"flatten_all incomplete after {attempts} polls; "
            f"open_positions={symbols}"
        )
