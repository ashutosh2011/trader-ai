"""Server-rendered parameter schemas for each registered strategy.

Single source of truth for the backtest form's "configure params" UI. The
schema is declarative — name, label, type, bounds, default, help text —
and the same data is exported as JSON for the browser so the dashboard
can render strategy-specific input controls without an extra request.

TRADEOFF: We keep the schema in a hand-written module (not auto-derived
from the strategy class) because the constructor signatures lack the
human-friendly labels, hints, and bounds the UI needs. A unit test
guards against drift by asserting every declared param name maps to a
real ``__init__`` keyword on the registered strategy class.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Literal

from strategies.registry import get_strategy

ParamType = Literal["int", "float"]


@dataclass(frozen=True)
class ParamSpec:
    """Schema for a single strategy parameter rendered as a number input."""

    name: str
    label: str
    type: ParamType
    default: int | float
    min: float
    max: float
    step: float
    help: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for the browser-side renderer."""
        return asdict(self)


@dataclass(frozen=True)
class StrategySchema:
    """Display-ready schema for one registered strategy."""

    id: str
    label: str
    summary: str
    params: tuple[ParamSpec, ...]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for the browser-side renderer."""
        return {
            "id": self.id,
            "label": self.label,
            "summary": self.summary,
            "params": [p.to_json() for p in self.params],
        }


# TRADEOFF: Reasonable defaults for min/max/step are picked once here
# rather than derived from the strategy. Periods bounded 1..200 (cover
# day-traders + longer trend templates), multipliers bounded 0.1..10.0,
# RSI thresholds bounded 1..99. These match the validations baked into
# the strategy constructors themselves.
STRATEGY_SCHEMAS: dict[str, StrategySchema] = {
    "ema_crossover": StrategySchema(
        id="ema_crossover",
        label="EMA crossover",
        summary="Fast/slow EMA cross with ATR-based stop and target.",
        params=(
            ParamSpec(
                name="fast_period",
                label="Fast period",
                type="int",
                default=12,
                min=1,
                max=200,
                step=1,
                help="Bars used for the faster EMA.",
            ),
            ParamSpec(
                name="slow_period",
                label="Slow period",
                type="int",
                default=26,
                min=2,
                max=200,
                step=1,
                help="Bars used for the slower EMA. Must exceed fast period.",
            ),
            ParamSpec(
                name="atr_period",
                label="ATR period",
                type="int",
                default=14,
                min=1,
                max=200,
                step=1,
                help="Lookback for ATR used to size stop/target.",
            ),
        ),
    ),
    "rsi_mean_revert": StrategySchema(
        id="rsi_mean_revert",
        label="RSI mean reversion",
        summary="RSI crosses through oversold/overbought with ATR stop/target.",
        params=(
            ParamSpec(
                name="rsi_period",
                label="RSI period",
                type="int",
                default=14,
                min=1,
                max=200,
                step=1,
                help="Bars used for the RSI calculation.",
            ),
            ParamSpec(
                name="oversold",
                label="Oversold threshold",
                type="float",
                default=30.0,
                min=1,
                max=99,
                step=1,
                help="RSI level below which a bullish cross goes long.",
            ),
            ParamSpec(
                name="overbought",
                label="Overbought threshold",
                type="float",
                default=70.0,
                min=1,
                max=99,
                step=1,
                help="RSI level above which a bearish cross goes short.",
            ),
            ParamSpec(
                name="atr_period",
                label="ATR period",
                type="int",
                default=14,
                min=1,
                max=200,
                step=1,
                help="Lookback for ATR used to size stop/target.",
            ),
            ParamSpec(
                name="stop_atr_mult",
                label="Stop ATR multiplier",
                type="float",
                default=1.0,
                min=0.1,
                max=10,
                step=0.1,
                help="Stop distance in ATR units.",
            ),
            ParamSpec(
                name="target_atr_mult",
                label="Target ATR multiplier",
                type="float",
                default=1.5,
                min=0.1,
                max=10,
                step=0.1,
                help="Target distance in ATR units.",
            ),
        ),
    ),
    "bbands_breakout": StrategySchema(
        id="bbands_breakout",
        label="Bollinger Bands breakout",
        summary="Long/short on close breaking outside the bands.",
        params=(
            ParamSpec(
                name="bb_period",
                label="Band period",
                type="int",
                default=20,
                min=1,
                max=200,
                step=1,
                help="Lookback for the moving average and standard deviation.",
            ),
            ParamSpec(
                name="bb_mult",
                label="Band multiplier",
                type="float",
                default=2.0,
                min=0.1,
                max=10,
                step=0.1,
                help="Standard-deviation multiplier for the upper/lower bands.",
            ),
            ParamSpec(
                name="atr_period",
                label="ATR period",
                type="int",
                default=14,
                min=1,
                max=200,
                step=1,
                help="Lookback for ATR used to size stop/target.",
            ),
            ParamSpec(
                name="stop_atr_mult",
                label="Stop ATR multiplier",
                type="float",
                default=1.0,
                min=0.1,
                max=10,
                step=0.1,
                help="Stop distance in ATR units.",
            ),
            ParamSpec(
                name="target_atr_mult",
                label="Target ATR multiplier",
                type="float",
                default=2.0,
                min=0.1,
                max=10,
                step=0.1,
                help="Target distance in ATR units.",
            ),
        ),
    ),
    "macd_trend": StrategySchema(
        id="macd_trend",
        label="MACD trend",
        summary="MACD signal-line cross filtered by histogram sign.",
        params=(
            ParamSpec(
                name="macd_fast",
                label="MACD fast",
                type="int",
                default=12,
                min=1,
                max=200,
                step=1,
                help="Fast EMA period inside the MACD line.",
            ),
            ParamSpec(
                name="macd_slow",
                label="MACD slow",
                type="int",
                default=26,
                min=2,
                max=200,
                step=1,
                help="Slow EMA period inside the MACD line.",
            ),
            ParamSpec(
                name="macd_signal",
                label="MACD signal",
                type="int",
                default=9,
                min=1,
                max=200,
                step=1,
                help="Signal-line EMA period applied to the MACD line.",
            ),
            ParamSpec(
                name="atr_period",
                label="ATR period",
                type="int",
                default=14,
                min=1,
                max=200,
                step=1,
                help="Lookback for ATR used to size stop/target.",
            ),
            ParamSpec(
                name="stop_atr_mult",
                label="Stop ATR multiplier",
                type="float",
                default=1.0,
                min=0.1,
                max=10,
                step=0.1,
                help="Stop distance in ATR units.",
            ),
            ParamSpec(
                name="target_atr_mult",
                label="Target ATR multiplier",
                type="float",
                default=2.0,
                min=0.1,
                max=10,
                step=0.1,
                help="Target distance in ATR units.",
            ),
        ),
    ),
    "supertrend_follow": StrategySchema(
        id="supertrend_follow",
        label="Supertrend follow",
        summary="Trade Supertrend direction flips with ATR stop/target.",
        params=(
            ParamSpec(
                name="st_period",
                label="Supertrend period",
                type="int",
                default=10,
                min=1,
                max=200,
                step=1,
                help="Bars used by the Supertrend ATR.",
            ),
            ParamSpec(
                name="st_multiplier",
                label="Supertrend multiplier",
                type="float",
                default=3.0,
                min=0.1,
                max=10,
                step=0.1,
                help="ATR multiplier defining the Supertrend bands.",
            ),
            ParamSpec(
                name="atr_period",
                label="ATR period",
                type="int",
                default=14,
                min=1,
                max=200,
                step=1,
                help="Independent ATR lookback used to size stop/target.",
            ),
            ParamSpec(
                name="stop_atr_mult",
                label="Stop ATR multiplier",
                type="float",
                default=1.0,
                min=0.1,
                max=10,
                step=0.1,
                help="Stop distance in ATR units.",
            ),
            ParamSpec(
                name="target_atr_mult",
                label="Target ATR multiplier",
                type="float",
                default=2.0,
                min=0.1,
                max=10,
                step=0.1,
                help="Target distance in ATR units.",
            ),
        ),
    ),
}


def get_schema(strategy_id: str) -> StrategySchema | None:
    """Return the schema for ``strategy_id`` or ``None`` when unknown."""
    return STRATEGY_SCHEMAS.get(strategy_id)


def all_schemas() -> list[StrategySchema]:
    """Return every registered schema sorted by display label."""
    return sorted(STRATEGY_SCHEMAS.values(), key=lambda s: s.label.lower())


def to_json_dict() -> dict[str, Any]:
    """Return ``{strategy_id: schema_json}`` for embedding in templates."""
    return {sid: schema.to_json() for sid, schema in STRATEGY_SCHEMAS.items()}


def strategy_param_keys(strategy_id: str) -> set[str]:
    """Return the kwargs accepted by ``strategy_id``'s constructor.

    Used by the API validator to reject unknown params before dispatching
    to the runner. Excludes ``self`` and ``symbol`` (the runner injects
    ``symbol`` itself).
    """
    cls = get_strategy(strategy_id)
    sig = inspect.signature(cls.__init__)
    return {name for name in sig.parameters if name not in {"self", "symbol"}}


__all__ = [
    "ParamSpec",
    "STRATEGY_SCHEMAS",
    "StrategySchema",
    "all_schemas",
    "get_schema",
    "strategy_param_keys",
    "to_json_dict",
]
