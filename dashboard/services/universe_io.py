"""Friendly CRUD for ``config/universe.yaml``.

Backs the ``/universe`` page so the operator never has to open the YAML
file in a text editor. Loads from :func:`screener.universe.load_universe`
(reuses the example fallback) and writes through Pydantic validation,
preserving a ``.bak`` of the previous file.

TRADEOFF: We don't preserve YAML comments because the universe file is
small and the dashboard reformats it into a uniform list-of-mappings.
This is a deliberate departure from ``config_io.save_yaml`` (which
keeps the operator's raw text) — the universe file has no need for
narrative comments.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from screener.universe import (
    DEFAULT_UNIVERSE_PATH,
    Universe,
    UniverseSymbol,
    load_universe,
)

logger = structlog.get_logger(__name__)

POPULAR_SEED_SYMBOLS: tuple[UniverseSymbol, ...] = (
    UniverseSymbol(symbol="RELIANCE", instrument_token=738561, exchange="NSE"),
    UniverseSymbol(symbol="INFY", instrument_token=408065, exchange="NSE"),
    UniverseSymbol(symbol="TCS", instrument_token=2953217, exchange="NSE"),
    UniverseSymbol(symbol="HDFCBANK", instrument_token=341249, exchange="NSE"),
    UniverseSymbol(symbol="ICICIBANK", instrument_token=1270529, exchange="NSE"),
    UniverseSymbol(symbol="SBIN", instrument_token=779521, exchange="NSE"),
    UniverseSymbol(symbol="BHARTIARTL", instrument_token=2714625, exchange="NSE"),
    UniverseSymbol(symbol="ITC", instrument_token=424961, exchange="NSE"),
    UniverseSymbol(symbol="LT", instrument_token=2939649, exchange="NSE"),
    UniverseSymbol(symbol="AXISBANK", instrument_token=1510401, exchange="NSE"),
    UniverseSymbol(symbol="TATAMOTORS", instrument_token=884737, exchange="NSE"),
    UniverseSymbol(symbol="BAJFINANCE", instrument_token=81153, exchange="NSE"),
    UniverseSymbol(symbol="KOTAKBANK", instrument_token=492033, exchange="NSE"),
    UniverseSymbol(symbol="HCLTECH", instrument_token=1850625, exchange="NSE"),
    UniverseSymbol(symbol="MARUTI", instrument_token=2815745, exchange="NSE"),
)


class UniverseIOError(ValueError):
    """Raised when the requested change violates the universe schema."""


class UniverseIO:
    """Read / write the screener universe YAML for the dashboard."""

    def __init__(self, *, universe_path: Path | None = None) -> None:
        """Construct the editor bound to a specific universe path.

        Args:
            universe_path: Override the target file. Defaults to
                :data:`screener.universe.DEFAULT_UNIVERSE_PATH`. Reads
                fall back to the bundled example via
                :func:`load_universe` when this file does not yet exist.
        """
        self._path = universe_path if universe_path is not None else DEFAULT_UNIVERSE_PATH

    @property
    def path(self) -> Path:
        """Target ``config/universe.yaml`` path."""
        return self._path

    def load_universe_editable(self) -> Universe:
        """Return the current universe, falling back to the example file.

        Raises:
            FileNotFoundError: When neither the user file nor the example
                exists (this is an installation-level error).
            UniverseIOError: When the on-disk file fails schema validation.
        """
        try:
            return load_universe(self._path if self._path.is_file() else None)
        except FileNotFoundError:
            raise
        except ValueError as exc:
            raise UniverseIOError(str(exc)) from exc

    def save_universe(self, symbols: list[UniverseSymbol]) -> Path:
        """Validate ``symbols`` and persist them to the universe file.

        The previous file (if any) is backed up to ``.bak`` before the
        new file lands. A YAML dump is used (not raw text) because the
        dashboard owns the file's format on this code path.

        Returns:
            The path that was written.

        Raises:
            UniverseIOError: When ``symbols`` is empty or duplicates a
                symbol, or when Pydantic rejects the shape.
        """
        if not symbols:
            msg = "universe must contain at least one symbol"
            raise UniverseIOError(msg)
        seen: set[str] = set()
        for entry in symbols:
            upper = entry.symbol.upper()
            if upper in seen:
                msg = f"duplicate symbol in universe: {entry.symbol}"
                raise UniverseIOError(msg)
            seen.add(upper)
        try:
            universe = Universe(symbols=list(symbols))
        except ValidationError as exc:
            raise UniverseIOError(str(exc)) from exc

        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.is_file():
            backup = self._path.with_name(self._path.name + ".bak")
            backup.write_bytes(self._path.read_bytes())
        payload = {
            "symbols": [
                {
                    "symbol": s.symbol,
                    "instrument_token": s.instrument_token,
                    "exchange": s.exchange,
                }
                for s in universe.symbols
            ]
        }
        text = yaml.safe_dump(payload, sort_keys=False)
        self._path.write_text(text, encoding="utf-8")
        logger.info(
            "dashboard_universe_saved",
            path=str(self._path),
            size=len(universe.symbols),
        )
        return self._path

    def add_symbol(
        self,
        *,
        symbol: str,
        exchange: str,
        instrument_token: int | None,
    ) -> Universe:
        """Append a new symbol and persist; returns the resulting universe."""
        symbols = self._current_symbols()
        if any(s.symbol.upper() == symbol.upper() for s in symbols):
            msg = f"symbol already in universe: {symbol}"
            raise UniverseIOError(msg)
        try:
            entry = UniverseSymbol(
                symbol=symbol.upper(),
                exchange=exchange,
                instrument_token=instrument_token,
            )
        except ValidationError as exc:
            raise UniverseIOError(str(exc)) from exc
        symbols.append(entry)
        self.save_universe(symbols)
        return Universe(symbols=symbols)

    def update_symbol(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        instrument_token: int | None = None,
    ) -> Universe:
        """Update an existing symbol's exchange / instrument_token in place."""
        symbols = self._current_symbols()
        upper = symbol.upper()
        updated: list[UniverseSymbol] = []
        found = False
        for entry in symbols:
            if entry.symbol.upper() == upper:
                found = True
                updates: dict[str, object] = {}
                if exchange is not None:
                    updates["exchange"] = exchange
                if instrument_token is not None:
                    updates["instrument_token"] = instrument_token
                updated.append(entry.model_copy(update=updates))
            else:
                updated.append(entry)
        if not found:
            msg = f"symbol not in universe: {symbol}"
            raise UniverseIOError(msg)
        self.save_universe(updated)
        return Universe(symbols=updated)

    def delete_symbol(self, symbol: str) -> Universe:
        """Remove ``symbol`` from the universe (case-insensitive)."""
        symbols = self._current_symbols()
        upper = symbol.upper()
        remaining = [s for s in symbols if s.symbol.upper() != upper]
        if len(remaining) == len(symbols):
            msg = f"symbol not in universe: {symbol}"
            raise UniverseIOError(msg)
        if not remaining:
            msg = "cannot delete the last symbol — add another first"
            raise UniverseIOError(msg)
        self.save_universe(remaining)
        return Universe(symbols=remaining)

    def seed_popular(self, *, limit: int = 15) -> int:
        """Append unseen symbols from :data:`POPULAR_SEED_SYMBOLS` (idempotent).

        Args:
            limit: Maximum number of new symbols to add.

        Returns:
            Count of symbols actually added (zero when everything already
            existed).
        """
        symbols = self._current_symbols() if self._path.is_file() else []
        existing = {s.symbol.upper() for s in symbols}
        to_add = [s for s in POPULAR_SEED_SYMBOLS if s.symbol.upper() not in existing]
        if limit > 0:
            to_add = to_add[:limit]
        if not to_add:
            return 0
        new_symbols = symbols + to_add
        self.save_universe(new_symbols)
        return len(to_add)

    def _current_symbols(self) -> list[UniverseSymbol]:
        if not self._path.is_file():
            try:
                return list(self.load_universe_editable().symbols)
            except (FileNotFoundError, UniverseIOError):
                return []
        return list(self.load_universe_editable().symbols)


__all__ = ["POPULAR_SEED_SYMBOLS", "UniverseIO", "UniverseIOError"]
