"""Risk manager: pre-trade checks and position sizing."""

import os
from datetime import datetime, time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from config.settings import DEFAULT_KILL_SWITCH_FILE, AppSettings, RiskConfig
from core.context import Context
from core.instrument import Instrument
from core.signal import IST, Signal

PositionSizingMode = Literal["atr_based", "fixed_pct"]


class RiskState(BaseModel):
    """Mutable portfolio risk snapshot for a trading session."""

    model_config = ConfigDict(frozen=False)

    account_equity: float = Field(gt=0)
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    open_positions: int = Field(ge=0, default=0)
    is_expiry_day: bool = False
    instrument_type: Literal["equity", "options", "future"] = "equity"

    @property
    def daily_pnl_mtm(self) -> float:
        """Realized + unrealized PnL for the current trading day."""
        return self.daily_realized_pnl + self.daily_unrealized_pnl


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
    """True when kill file exists or env var is ``1``.

    TRADEOFF: ``kill_file`` defaults to the tradebot-root absolute path so the
    kill switch is detected regardless of the caller's CWD. Pass an explicit
    path for tests or non-default deployments.
    """
    path = kill_file if kill_file is not None else DEFAULT_KILL_SWITCH_FILE
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

    def pre_check(
        self,
        signal: Signal,
        ctx: Context,
        state: RiskState,
        *,
        instrument: Instrument | None = None,
    ) -> RiskDecision:
        """Validate whether a new entry is allowed.

        Args:
            signal: Candidate entry signal.
            ctx: Strategy context for the current bar.
            state: Live portfolio risk snapshot. The daily-loss cap is enforced
                against the mark-to-market PnL (realized + unrealized).
            instrument: Optional instrument metadata. Accepted for forward
                compatibility; not used by current pre-check checks but the
                signature stays uniform with :meth:`size` / :meth:`post_check`.

        Returns:
            A :class:`RiskDecision` with ``approved`` and a short ``reason``.
        """
        del instrument  # currently unused but kept for API uniformity
        if is_kill_switch_active(
            kill_file=self._settings.kill_switch_file,
            env_var=self._settings.kill_switch_env,
        ):
            return RiskDecision(approved=False, reason="kill_switch_active")

        if state.open_positions >= self._risk.max_open_positions:
            return RiskDecision(approved=False, reason="max_open_positions")

        daily_cap = state.account_equity * (self._risk.daily_loss_cap_pct / 100.0)
        if state.daily_pnl_mtm <= -daily_cap:
            return RiskDecision(approved=False, reason="daily_loss_cap_mtm")

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
        *,
        instrument: Instrument | None = None,
        size_multiplier: float = 1.0,
    ) -> int:
        """Compute order quantity from risk config, analyst multiplier, and lot size.

        Args:
            signal: Candidate entry signal.
            ctx: Strategy context for the current bar.
            state: Live portfolio risk snapshot.
            instrument: Optional instrument metadata. When supplied, the
                quantity is floored to whole exchange lots; defaults to an
                equity-like ``lot_size=1`` when ``None``.
            size_multiplier: Analyst scaling factor in ``[0.0, 1.0]``.

        Returns:
            The order quantity. **May be zero** when the configured risk
            budget is below the instrument's lot size or when the analyst
            multiplier shrinks the base quantity below one lot. Callers
            should rely on :meth:`post_check` to reject ``qty == 0`` rather
            than silently floor to one.
        """
        del ctx  # unused; retained for API uniformity
        multiplier = max(0.0, min(1.0, size_multiplier))
        base_qty = self._base_qty(signal, state)
        lot_size = instrument.lot_size if instrument is not None else 1
        # TRADEOFF: apply the analyst multiplier BEFORE flooring to lot size
        # rather than after. This keeps the analyst's "shrink" semantics
        # honest — multiplier=0.0 yields qty=0 (no silent floor to one lot)
        # and multipliers between two lots round down to the safer side.
        scaled = int(base_qty * multiplier)
        return (scaled // lot_size) * lot_size

    def post_check(
        self,
        signal: Signal,
        ctx: Context,
        state: RiskState,
        qty: int,
        *,
        instrument: Instrument | None = None,
    ) -> RiskDecision:
        """Validate the post-sized order against per-trade risk caps.

        Args:
            signal: Candidate entry signal.
            ctx: Strategy context for the current bar.
            state: Live portfolio risk snapshot.
            qty: Quantity returned by :meth:`size`. ``0`` is rejected with
                ``qty_zero_below_lot_size`` if the risk-budget sizing already
                fell below a single lot, or ``qty_zero_from_shrink`` when the
                analyst multiplier collapsed an otherwise-sufficient size.
            instrument: Optional instrument metadata; defaults to lot_size=1.

        Returns:
            A :class:`RiskDecision` with ``approved`` and a short ``reason``.
        """
        del ctx  # unused; retained for API uniformity
        lot_size = instrument.lot_size if instrument is not None else 1
        if qty <= 0:
            base_qty = self._base_qty(signal, state)
            if base_qty < lot_size:
                return RiskDecision(approved=False, reason="qty_zero_below_lot_size")
            return RiskDecision(approved=False, reason="qty_zero_from_shrink")

        max_risk = state.account_equity * (self._risk.max_loss_per_trade_pct / 100.0)
        per_share_risk = abs(signal.entry - signal.stop_loss)
        if per_share_risk <= 0:
            return RiskDecision(approved=False, reason="invalid_stop_distance")
        trade_risk = per_share_risk * qty
        if trade_risk > max_risk * 1.01:
            return RiskDecision(approved=False, reason="trade_risk_exceeds_cap")

        # TRADEOFF: For long-options the per-share-stop check above can pass
        # easily because premiums are small; without a notional cap we could
        # still deploy too much capital into a single decay-prone position.
        # We cap total *premium* (entry * qty * lot_size) at a configured
        # percentage of equity. Equity (instrument_type="equity") is excluded
        # because the stop-distance cap already governs its capital use.
        if instrument is not None and instrument.instrument_type == "option":
            notional = signal.entry * qty * instrument.lot_size
            cap = state.account_equity * (self._risk.max_premium_per_trade_pct / 100.0)
            if notional > cap:
                return RiskDecision(approved=False, reason="notional_cap_exceeded")

        return RiskDecision(approved=True, reason="ok")

    def _base_qty(self, signal: Signal, state: RiskState) -> int:
        max_risk_amount = state.account_equity * (self._risk.max_loss_per_trade_pct / 100.0)
        per_share_risk = abs(signal.entry - signal.stop_loss)
        if per_share_risk <= 0:
            return 0
        if self._risk.position_sizing == "fixed_pct":
            notional = state.account_equity * (self._risk.fixed_position_pct / 100.0)
            return max(0, int(notional / signal.entry))
        return max(0, int(max_risk_amount / per_share_risk))
