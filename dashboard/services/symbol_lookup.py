"""Read-only symbol autocomplete backed by the screener universe.

The backtest form (and any future page that picks a symbol) calls this
service through ``/api/symbols/search``. The data source is the same
``config/universe.yaml`` (with example fallback) that the screener
already consumes; we never reach for Kite's instruments dump.

TRADEOFF: Ranking is intentionally simple — exact match first, then
prefix match, then substring, then alphabetical. The dashboard universe
is small (tens of symbols), so a richer fuzzy match would buy nothing
but extra surface area.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

from screener.universe import Universe, UniverseSymbol, load_universe

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SymbolEntry:
    """One symbol surfaced by the picker."""

    symbol: str
    exchange: str
    instrument_token: int | None
    display_label: str

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serialisable dict (used by the API)."""
        return asdict(self)


class SymbolLookupService:
    """Search the screener universe by partial symbol match.

    The service rereads the universe file on every call so edits via
    ``/universe`` are visible immediately. The universe is tiny (tens
    of symbols), so the I/O cost is negligible.
    """

    def __init__(self, *, universe_path: Path | None = None) -> None:
        """Construct the service bound to a specific universe file.

        Args:
            universe_path: Optional override; ``None`` lets
                :func:`load_universe` pick between
                ``config/universe.yaml`` and the bundled example.
        """
        self._path = universe_path

    def _load(self) -> Universe | None:
        try:
            return load_universe(self._path)
        except FileNotFoundError:
            return None
        except ValueError as exc:
            logger.warning("dashboard_symbol_lookup_invalid", error=str(exc))
            return None

    def list_symbols(self) -> list[SymbolEntry]:
        """Return every entry in the universe (alphabetical)."""
        universe = self._load()
        if universe is None:
            return []
        entries = [_to_entry(s) for s in universe.symbols]
        entries.sort(key=lambda e: e.symbol.upper())
        return entries

    def find_symbol(self, query: str) -> SymbolEntry | None:
        """Return the single entry whose symbol matches ``query`` (case-insensitive)."""
        target = query.strip().upper()
        if not target:
            return None
        for entry in self.list_symbols():
            if entry.symbol.upper() == target:
                return entry
        return None

    def search(self, query: str, *, limit: int = 20) -> list[SymbolEntry]:
        """Rank entries against ``query`` and return up to ``limit`` matches.

        Args:
            query: Free-text fragment. Empty string returns the first
                ``limit`` alphabetical entries (useful for "open the
                dropdown on focus" UX).
            limit: Maximum number of entries to return.

        Returns:
            Ordered: exact (case-insensitive) match → prefix match →
            substring match → alphabetical fallback.
        """
        all_entries = self.list_symbols()
        if limit <= 0:
            return []
        target = query.strip().upper()
        if not target:
            return all_entries[:limit]

        exact: list[SymbolEntry] = []
        prefix: list[SymbolEntry] = []
        substring: list[SymbolEntry] = []
        for entry in all_entries:
            upper = entry.symbol.upper()
            if upper == target:
                exact.append(entry)
            elif upper.startswith(target):
                prefix.append(entry)
            elif target in upper:
                substring.append(entry)
        merged = exact + prefix + substring
        return merged[:limit]


def _to_entry(symbol: UniverseSymbol) -> SymbolEntry:
    token_part = (
        str(symbol.instrument_token) if symbol.instrument_token is not None else "no token"
    )
    return SymbolEntry(
        symbol=symbol.symbol,
        exchange=symbol.exchange,
        instrument_token=symbol.instrument_token,
        display_label=f"{symbol.symbol} — {symbol.exchange} — {token_part}",
    )


__all__ = ["SymbolEntry", "SymbolLookupService"]
