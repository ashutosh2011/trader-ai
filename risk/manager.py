"""Risk manager: pre-trade checks and position sizing."""

import os
from datetime import datetime, time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from config.settings import AppSettings, RiskConfig
from core.context import Context
from core.signal import IST, Signal

PositionSizingMode = Literal["atr_based", "fixed_pct"]


class RiskState(BaseModel):
    """Mutable portfolio risk snapshot for a trading session."""

    model_config = ConfigDict(frozen=False)

    account_equity: float = Field(gt=0)
    daily_realized_pnl: float = 0.0
    open_positions: int = Field(ge=0, default=0)
    is_expiry_day: bool = False
    instrument_type: Literal["equity", "options", "future"] = "equity"


class RiskDecision(BaseModel):
    """Outcome of a risk pre-check or post-check."""

    model_config = ConfigDict(frozen=True)

    approved: bool
    reason: str = ""


def parse_time_window(window: str) -> tuple[time, time]:
    """Parse ``HH:MM-HH:MM`` into start/end :class:`time` objects."""
    parts = window.strip().split("-")
    if len(parts) != 2:
        msg = f"invalid time window: {window!r}"
        raise ValueError(msg)
    start = _parse_hhmm(parts[0].strip())
    end = _parse_hhmm(parts[1].strip())
    return start, end


def _parse_hhmm(value: str) -> time:
    hour_str, minute_str = value.split(":")
    return time(hour=int(hour_str), minute=int(minute_str))


def is_in_no_trade_window(ts: datetime, windows: list[str]) -> bool:
    """Return True if ``ts`` (IST) falls inside any no-trade window."""
    local = ts.astimezone(IST)
    current = local.time()
    for window in windows:
        start, end = parse_time_window(window)
        if start <= end:
            if start <= current <= end:
                return True
        elif current >= start or current <= end:
            return True
    return False


def is_kill_switch_active(
    *,
    kill_file: Path | None = None,
    env_var: str = "KILL_SWITCH",
) -> bool:
    """True when kill file exists or env var is ``1``."""
    path = kill_file or Path("KILL")
    if path.is_file():
        return True
    return os.environ.get(env_var, "0").strip() in {"1", "true", "TRUE", "yes", "YES"}


class RiskManager:
    """Enforces risk rules and computes position size."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or AppSettings()
        self._risk: RiskConfig = self._settings.risk

    @property
    def config(self) -> RiskConfig:
        return self._risk

    def pre_check(self, signal: Signal, ctx: Context, state: RiskState) -> RiskDecision:
        """Validate whether a new entry is allowed."""
        if is_kill_switch_active(
            kill_file=self._settings.kill_switch_file,
            env_var=self._settings.kill_switch_env,
        ):
            return RiskDecision(approved=False, reason="kill_switch_active")

        if state.open_positions >= self._risk.max_open_positions:
            return RiskDecision(approved=False, reason="max_open_positions")

        daily_cap = state.account_equity * (self._risk.daily_loss_cap_pct / 100.0)
        if state.daily_realized_pnl <= -daily_cap:
            return RiskDecision(approved=False, reason="daily_loss_cap")

        if is_in_no_trade_window(signal.ts, self._risk.no_trade_windows):
            return RiskDecision(approved=False, reason="no_trade_window")

        if state.instrument_type == "options" and state.is_expiry_day:
            block_after = _parse_hhmm(
                self._risk.expiry_day_rules.block_new_options_entries_after
            )
            if signal.ts.astimezone(IST).time() >= block_after:
                return RiskDecision(approved=False, reason="expiry_day_entry_blocked")

        return RiskDecision(approved=True, reason="ok")

    def size(
        self,
        signal: Signal,
        ctx: Context,
        state: RiskState,
        size_multiplier: float = 1.0,
    ) -> int:
        """Compute order quantity from risk config and analyst multiplier."""
        multiplier = max(0.0, min(1.0, size_multiplier))
        base_qty = self._base_qty(signal, state)
        qty = max(1, int(base_qty * multiplier))
        return qty

    def post_check(
        self,
        signal: Signal,
        ctx: Context,
        state: RiskState,
        qty: int,
    ) -> RiskDecision:
        """Optional validation after sizing."""
        if qty < 1:
            return RiskDecision(approved=False, reason="qty_below_minimum")
        max_risk = state.account_equity * (self._risk.max_loss_per_trade_pct / 100.0)
        per_share_risk = abs(signal.entry - signal.stop_loss)
        if per_share_risk <= 0:
            return RiskDecision(approved=False, reason="invalid_stop_distance")
        trade_risk = per_share_risk * qty
        if trade_risk > max_risk * 1.01:
            return RiskDecision(approved=False, reason="trade_risk_exceeds_cap")
        return RiskDecision(approved=True, reason="ok")

    def _base_qty(self, signal: Signal, state: RiskState) -> int:
        max_risk_amount = state.account_equity * (self._risk.max_loss_per_trade_pct / 100.0)
        per_share_risk = abs(signal.entry - signal.stop_loss)
        if per_share_risk <= 0:
            return 1
        if self._risk.position_sizing == "fixed_pct":
            notional = state.account_equity * (self._risk.fixed_position_pct / 100.0)
            return max(1, int(notional / signal.entry))
        return max(1, int(max_risk_amount / per_share_risk))
