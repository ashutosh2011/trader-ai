from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from config.settings import PaperConfig
from core.signal import Signal
from execution.broker import deterministic_client_order_id
from execution.paper import PaperBroker

IST = ZoneInfo("Asia/Kolkata")


def _signal(**kwargs: object) -> Signal:
    defaults = {
        "symbol": "SYNTH",
        "side": "BUY",
        "entry": 100.0,
        "stop_loss": 99.0,
        "target": 102.0,
        "timeframe": "1m",
        "strategy_id": "ema_crossover",
        "reasons": ["test"],
        "indicator_snapshot": {},
        "confidence": 0.5,
        "ts": datetime(2024, 1, 1, 10, 0, tzinfo=IST),
    }
    defaults.update(kwargs)
    return Signal(**defaults)  # type: ignore[arg-type]


def test_deterministic_client_order_id() -> None:
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    a = deterministic_client_order_id("s1", ts, "X")
    b = deterministic_client_order_id("s1", ts, "X")
    assert a == b
    assert a.startswith("tb-")


def test_bracket_order_slippage_and_position() -> None:
    broker = PaperBroker(paper_config=PaperConfig(slippage_bps=10.0, account_equity=50_000.0))
    result = broker.place_bracket_order(_signal(), qty=2)
    assert result.status == "FILLED"
    assert result.fill_price == pytest.approx(100.0 * 1.001)
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].qty == 2


def test_duplicate_position_rejected() -> None:
    broker = PaperBroker()
    broker.place_bracket_order(_signal(), qty=1)
    second = broker.place_bracket_order(_signal(), qty=1)
    assert second.status == "REJECTED"


def test_bar_exit_stop_before_target() -> None:
    broker = PaperBroker()
    broker.place_bracket_order(_signal(entry=100.0, stop_loss=99.0, target=101.0), qty=1)
    price, reason = broker.check_bar_exits("SYNTH", high=101.5, low=98.5)
    assert price == 99.0
    assert reason == "stop_loss"
    pnl = broker.close_position("SYNTH", price, reason)
    assert pnl == pytest.approx(-1.05)  # entry 100.05 with default 5bps slippage, exit 99


def test_flatten_all() -> None:
    broker = PaperBroker()
    broker.place_bracket_order(_signal(), qty=1)
    broker.flatten_all()
    assert broker.get_positions() == []


def test_mark_to_market_no_positions() -> None:
    broker = PaperBroker(paper_config=PaperConfig(slippage_bps=0.0, account_equity=50_000.0))
    assert broker.mark_to_market({}) == pytest.approx(0.0)
    assert broker.mark_to_market({"SYNTH": 123.45}) == pytest.approx(0.0)


def test_mark_to_market_long_position_unrealized() -> None:
    broker = PaperBroker(paper_config=PaperConfig(slippage_bps=0.0, account_equity=50_000.0))
    broker.place_bracket_order(
        _signal(side="BUY", entry=100.0, stop_loss=99.0, target=102.0),
        qty=1,
    )
    assert broker.mark_to_market({"SYNTH": 105.0}) == pytest.approx(5.0)


def test_mark_to_market_short_position_unrealized() -> None:
    broker = PaperBroker(paper_config=PaperConfig(slippage_bps=0.0, account_equity=50_000.0))
    broker.place_bracket_order(
        _signal(side="SELL", entry=100.0, stop_loss=101.0, target=98.0),
        qty=1,
    )
    assert broker.mark_to_market({"SYNTH": 95.0}) == pytest.approx(5.0)


def test_mark_to_market_missing_symbol_is_zero() -> None:
    broker = PaperBroker(paper_config=PaperConfig(slippage_bps=0.0, account_equity=50_000.0))
    broker.place_bracket_order(
        _signal(side="BUY", entry=100.0, stop_loss=99.0, target=102.0),
        qty=1,
    )
    assert broker.mark_to_market({"OTHER": 200.0}) == pytest.approx(0.0)
