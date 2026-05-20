"""Market hours, holidays, and F&O expiry scheduling (IST)."""

from __future__ import annotations

from datetime import date, datetime, time

from pydantic import BaseModel, ConfigDict

from config.settings import AppSettings, SchedulerConfig
from core.signal import IST

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class ScheduleDecision(BaseModel):
    """Whether trading activity is allowed at a timestamp."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str = ""


def parse_hhmm(value: str) -> time:
    """Parse ``HH:MM`` into a :class:`time`."""
    hour_str, minute_str = value.split(":")
    return time(hour=int(hour_str), minute=int(minute_str))


class MarketScheduler:
    """NSE session rules: market hours, holidays, expiry windows."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        app = settings or AppSettings()
        self._config: SchedulerConfig = app.scheduler
        self._risk_windows = app.risk.no_trade_windows
        self._open = parse_hhmm(self._config.market_open)
        self._close = parse_hhmm(self._config.market_close)
        self._expiry_block = parse_hhmm(self._config.block_fno_on_expiry_after)
        self._holidays = {_parse_date(d) for d in self._config.holiday_skip_dates}

    def is_holiday(self, day: date) -> bool:
        """Return True when ``day`` is in the configured skip list."""
        return day in self._holidays

    def is_market_open(self, ts: datetime) -> bool:
        """Return True during NSE cash session (weekday, not holiday)."""
        local = ts.astimezone(IST)
        if local.weekday() >= 5:
            return False
        if self.is_holiday(local.date()):
            return False
        current = local.time()
        return self._open <= current <= self._close

    def is_expiry_day(self, ts: datetime) -> bool:
        """Return True on configured weekly expiry weekday (default Thursday)."""
        return ts.astimezone(IST).weekday() == self._config.expiry_weekday

    def should_process_bar(
        self,
        ts: datetime,
        *,
        instrument_type: str = "equity",
    ) -> ScheduleDecision:
        """Gate bar processing: market hours, holidays, expiry F&O rules."""
        if not self.is_market_open(ts):
            return ScheduleDecision(allowed=False, reason="market_closed")

        from risk.manager import is_in_no_trade_window

        if is_in_no_trade_window(ts, self._risk_windows):
            return ScheduleDecision(allowed=False, reason="no_trade_window")

        if instrument_type in {"options", "future"} and self.is_expiry_day(ts):
            local = ts.astimezone(IST)
            if local.time() >= self._expiry_block:
                return ScheduleDecision(allowed=False, reason="expiry_day_fno_blocked")

        return ScheduleDecision(allowed=True, reason="ok")


def _parse_date(value: str) -> date:
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))
