"""Universe loader.

The screener evaluates a fixed list of symbols loaded from a YAML file.
``config/universe.yaml`` is user-local (gitignored); if missing we fall
back to ``config/universe.example.yaml`` which ships with the repo.

TRADEOFF: ``instrument_token`` is optional because not every user has a
ready Kite instruments dump. The runner only enables Kite on-demand
fetch when a token is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = structlog.get_logger(__name__)

TRADEBOT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_UNIVERSE_PATH: Path = TRADEBOT_ROOT / "config" / "universe.yaml"
FALLBACK_UNIVERSE_PATH: Path = TRADEBOT_ROOT / "config" / "universe.example.yaml"


class UniverseSymbol(BaseModel):
    """One symbol entry in the universe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    instrument_token: int | None = None
    exchange: str = "NSE"

    @field_validator("symbol")
    @classmethod
    def symbol_non_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "symbol must be non-empty"
            raise ValueError(msg)
        return value


class Universe(BaseModel):
    """Container for the screener's symbol list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbols: list[UniverseSymbol] = Field(min_length=1)


def load_universe(path: Path | None = None) -> Universe:
    """Load the screener universe.

    Args:
        path: Explicit YAML path. When ``None``, tries
            ``config/universe.yaml`` first, then
            ``config/universe.example.yaml``.

    Returns:
        Parsed :class:`Universe`.

    Raises:
        FileNotFoundError: When neither ``path`` nor either default
            location exists.
        ValueError: When the YAML is structurally invalid (not a mapping,
            missing ``symbols`` key, schema violations).
    """
    candidates: list[Path] = (
        [path] if path is not None else [DEFAULT_UNIVERSE_PATH, FALLBACK_UNIVERSE_PATH]
    )

    chosen: Path | None = None
    for candidate in candidates:
        if candidate.is_file():
            chosen = candidate
            break

    if chosen is None:
        tried = ", ".join(str(c) for c in candidates)
        msg = (
            f"universe file not found (tried: {tried}). "
            f"Copy {FALLBACK_UNIVERSE_PATH.name} to {DEFAULT_UNIVERSE_PATH.name} "
            "or pass an explicit path."
        )
        raise FileNotFoundError(msg)

    raw = chosen.read_text(encoding="utf-8")
    try:
        data: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"failed to parse {chosen}: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(data, dict):
        msg = f"{chosen} must be a YAML mapping at the top level"
        raise ValueError(msg)

    try:
        universe = Universe.model_validate(data)
    except ValidationError as exc:
        msg = f"invalid universe schema in {chosen}: {exc}"
        raise ValueError(msg) from exc

    logger.debug(
        "screener_universe_loaded",
        path=str(chosen),
        size=len(universe.symbols),
    )
    return universe


__all__ = [
    "DEFAULT_UNIVERSE_PATH",
    "FALLBACK_UNIVERSE_PATH",
    "Universe",
    "UniverseSymbol",
    "load_universe",
]
