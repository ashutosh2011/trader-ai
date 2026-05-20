from datetime import date

import pytest

from core.instrument import Instrument


def test_nse_equity() -> None:
    inst = Instrument.nse_equity("RELIANCE")
    assert inst.segment == "EQ"
    assert inst.instrument_type == "equity"


def test_nse_future() -> None:
    inst = Instrument.nse_future(
        "NIFTY24MAYFUT",
        date(2024, 5, 30),
        underlying="NIFTY",
        lot_size=50,
    )
    assert inst.instrument_type == "future"
    assert inst.segment == "FO"


def test_nse_option() -> None:
    inst = Instrument.nse_option(
        "NIFTY24MAY24000CE",
        date(2024, 5, 30),
        24000.0,
        "CE",
        underlying="NIFTY",
        lot_size=50,
    )
    assert inst.option_type == "CE"


def test_equity_rejects_fo_fields() -> None:
    with pytest.raises(ValueError):
        Instrument(
            symbol="X",
            segment="EQ",
            instrument_type="equity",
            expiry=date(2024, 1, 1),
        )


def test_round_qty() -> None:
    inst = Instrument.nse_future("F", date(2024, 5, 30), underlying="N", lot_size=50)
    assert inst.round_qty(30) == 50
    assert inst.round_qty(75) == 100
