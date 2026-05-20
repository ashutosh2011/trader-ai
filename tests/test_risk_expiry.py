from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import AppSettings, RiskConfig
from core.context import Context
from risk.manager import RiskManager, RiskState
from tests.test_risk_manager import _signal

IST = ZoneInfo("Asia/Kolkata")


def test_expiry_day_options_block() -> None:
    settings = AppSettings(
        risk=RiskConfig(
            expiry_day_rules={"block_new_options_entries_after": "14:30"}  # type: ignore[arg-type]
        )
    )
    rm = RiskManager(settings)
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 15, 0, tzinfo=IST)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000.0],
        }
    )
    ctx = Context(
        symbol="OPT",
        bars=frame,
        bar_index=0,
        timestamp=frame["timestamp"].iloc[0].to_pydatetime(),
        timeframe="1m",
    )
    ts = datetime(2024, 1, 1, 15, 0, tzinfo=IST)
    state = RiskState(
        account_equity=100_000.0,
        is_expiry_day=True,
        instrument_type="options",
    )
    decision = rm.pre_check(_signal(ts), ctx, state)
    assert not decision.approved
    assert decision.reason == "expiry_day_entry_blocked"
