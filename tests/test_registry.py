import indicators  # noqa: F401 — register builtins
from indicators.builtin import (
    ATR,
    EMA,
    MACD,
    RSI,
    SMA,
    VWAP,
    BBands,
    Supertrend,
)
from indicators.registry import get_indicator, list_indicators


def test_all_builtins_registered() -> None:
    names = list_indicators()
    for expected in ("ema", "sma", "rsi", "macd", "atr", "vwap", "supertrend", "bbands"):
        assert expected in names
    assert get_indicator("ema") is EMA
    assert get_indicator("sma") is SMA
    assert get_indicator("rsi") is RSI
    assert get_indicator("macd") is MACD
    assert get_indicator("atr") is ATR
    assert get_indicator("vwap") is VWAP
    assert get_indicator("supertrend") is Supertrend
    assert get_indicator("bbands") is BBands
