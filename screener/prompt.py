"""LLM screener prompt construction.

The prompt instructs the model to emit a JSON object matching
:class:`screener.schema.ScreenerFormula` and nothing else. The prompt
includes:

* the indicator whitelist (so the LLM doesn't invent names);
* a description of each filter type;
* two example formulas (long mean-reversion + short trend-follow);
* explicit instructions to choose timeframe, side bias, and 2-5 filters.

TRADEOFF: We inline the schema as JSON rather than passing
``ScreenerFormula.model_json_schema()`` because the structured schema is
verbose and brittle across pydantic versions; a hand-written summary is
more LLM-friendly and easier to keep in sync with the docstring.
"""

from __future__ import annotations

from datetime import datetime
from textwrap import dedent

from pydantic import BaseModel, ConfigDict, field_validator

from screener.registry import whitelist_names
from screener.universe import Universe

PROMPT_VERSION = "v1"


class MarketContext(BaseModel):
    """Free-form context passed to the LLM screener.

    The dashboard fills this in from operator input; nothing in the
    schema is automatically harvested from market data. Keep the text
    short — the model uses it as a hint, not ground truth.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: datetime
    recent_index_summary: str
    notes: str = ""

    @field_validator("as_of")
    @classmethod
    def as_of_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "MarketContext.as_of must be timezone-aware"
            raise ValueError(msg)
        return value


_INDICATOR_DESCRIPTIONS: dict[str, str] = {
    "rsi": "Relative Strength Index (Wilder). Params: period (int, default 14).",
    "sma": "Simple moving average on close. Params: period (int, required).",
    "ema": "Exponential moving average on close. Params: span (int, required).",
    "atr": "Average True Range. Params: period (int, default 14).",
    "close": "Last close price. No params.",
    "open": "Last open price. No params.",
    "high": "Last high price. No params.",
    "low": "Last low price. No params.",
    "macd_line": "MACD line. Params: fast (default 12), slow (default 26), signal (default 9).",
    "macd_signal": "MACD signal line. Same params as macd_line.",
    "macd_hist": "MACD histogram (line - signal). Same params as macd_line.",
    "bb_upper": "Bollinger upper band. Params: period (default 20), mult (float, default 2.0).",
    "bb_middle": "Bollinger middle band (SMA). Same params as bb_upper.",
    "bb_lower": "Bollinger lower band. Same params as bb_upper.",
    "supertrend_dir": (
        "Supertrend direction (+1 uptrend, -1 downtrend). "
        "Params: period (default 10), multiplier (float, default 3.0)."
    ),
}


_EXAMPLE_FORMULAS: str = dedent(
    """\
    Example 1 — long mean-revert (daily oversold reclaim):
    {
      "name": "Oversold reclaim",
      "timeframe": "day",
      "side_bias": "long",
      "rationale": "Range-bound regime; buy oversold dips above the 50-day SMA.",
      "filters": [
        {"type": "indicator", "indicator": "rsi",
         "params": {"period": 14}, "op": "<", "value": 35.0},
        {"type": "indicator", "indicator": "close", "op": ">",
         "compare_to": {"indicator": "sma", "params": {"period": 50}}},
        {"type": "volume", "op": ">", "value_x_avg": 1.2, "avg_window": 20}
      ]
    }

    Example 2 — short trend-follow (5m breakdowns):
    {
      "name": "5m breakdown short",
      "timeframe": "5m",
      "side_bias": "short",
      "rationale": "Downtrend regime; short stocks losing the 20-period EMA on volume.",
      "filters": [
        {"type": "indicator", "indicator": "close", "op": "<",
         "compare_to": {"indicator": "ema", "params": {"span": 20}}},
        {"type": "price_change", "window": 5, "op": "<", "value_pct": -1.5},
        {"type": "volume", "op": ">", "value_x_avg": 1.5, "avg_window": 20}
      ]
    }
    """
)


def build_screener_prompt(market_context: MarketContext, universe: Universe) -> str:
    """Build the prompt for the LLM screener.

    Args:
        market_context: Free-form regime + notes input.
        universe: Symbols under consideration (listed verbatim).

    Returns:
        A prompt string the LLM should respond to with a single JSON
        object matching :class:`screener.schema.ScreenerFormula`.
    """
    indicator_block = "\n".join(
        f"  - {name}: {_INDICATOR_DESCRIPTIONS[name]}" for name in whitelist_names()
    )
    universe_symbols = ", ".join(item.symbol for item in universe.symbols)
    return dedent(
        f"""\
        You are a trading screener (prompt {PROMPT_VERSION}). Decide what kind of
        trade setup is suitable for the *current market regime* and return a
        SINGLE JSON object describing a filter formula. Output nothing else —
        no commentary, no markdown — only one JSON object.

        Reason about:
          1. The regime (trending, ranging, choppy) implied by the context.
          2. The right timeframe to look at: "day" (end-of-day swing setups)
             or "5m" (intraday breakouts/breakdowns).
          3. A side bias: "long", "short", or "both".
          4. 2–5 concrete filters with thresholds (no placeholders).

        Hard rules:
          * Use ONLY indicators from the whitelist below. Any other name will
            be rejected and the screener will fall back to a default formula.
          * All filters are AND-combined (no OR/NOT in v1).
          * Each indicator filter must have EITHER a numeric "value" OR a
            "compare_to" object — never both, never neither.
          * compare_to may not nest a compare_to of its own.

        Indicator whitelist:
        {indicator_block}

        Filter types:
          * type "indicator": compares an indicator's latest value to a
            constant or another indicator. Keys: indicator, params, op,
            and exactly one of value | compare_to.
          * type "volume": tests last-bar volume against either an absolute
            "value" OR "value_x_avg" multiple of a rolling mean over
            "avg_window" bars.
          * type "price_change": percent change over the last "window"
            closed bars compared to "value_pct".

        Allowed comparison ops: >, <, >=, <=, ==.

        JSON schema (exact keys):
          {{
            "name": str,                 # short human-readable name
            "timeframe": "day" | "5m",
            "side_bias": "long" | "short" | "both",
            "rationale": str,            # 1-3 sentences of reasoning
            "filters": [Filter, ...]     # 2-5 entries
          }}

        {_EXAMPLE_FORMULAS}

        Market context:
          as_of: {market_context.as_of.isoformat()}
          recent_index_summary: {market_context.recent_index_summary}
          notes: {market_context.notes}

        Universe under consideration ({len(universe.symbols)} symbols):
          {universe_symbols}

        Return ONLY the JSON object describing the formula.
        """
    )


__all__ = ["PROMPT_VERSION", "MarketContext", "build_screener_prompt"]
