from datetime import date, datetime
from zoneinfo import ZoneInfo

from config.settings import AppSettings, SchedulerConfig
from orchestrator.scheduler import MarketScheduler

IST = ZoneInfo("Asia/Kolkata")


def test_market_open_weekday() -> None:
    sched = MarketScheduler(AppSettings())
    ts = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
    assert sched.is_market_open(ts)


def test_market_closed_weekend() -> None:
    sched = MarketScheduler(AppSettings())
    ts = datetime(2024, 1, 6, 10, 0, tzinfo=IST)
    assert not sched.is_market_open(ts)


def test_holiday_skip() -> None:
    settings = AppSettings(
        scheduler=SchedulerConfig(holiday_skip_dates=["2024-01-02"]),
    )
    sched = MarketScheduler(settings)
    assert sched.is_holiday(date(2024, 1, 2))
    ts = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
    assert not sched.is_market_open(ts)


def test_expiry_day_thursday() -> None:
    sched = MarketScheduler(AppSettings())
    ts = datetime(2024, 1, 4, 10, 0, tzinfo=IST)
    assert sched.is_expiry_day(ts)


def test_should_process_bar_equity() -> None:
    sched = MarketScheduler(AppSettings())
    ts = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
    decision = sched.should_process_bar(ts, instrument_type="equity")
    assert decision.allowed
