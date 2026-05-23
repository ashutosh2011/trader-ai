"""Read-only symbol autocomplete backed by the cached Kite instruments.

The backtest form (and any future page that picks a symbol) calls this
service through ``/api/symbols/search``. The data source is now the
``instruments`` DuckDB table populated from
:py:meth:`kiteconnect.KiteConnect.instruments` via
:class:`InstrumentsService`. The legacy ``config/universe.yaml`` source
has been retired from this module — it remains a watchlist concept used
by other pages (the screener / universe editor) which the operator can
ignore for backtests.

TRADEOFF: We thinly wrap :class:`InstrumentsService` rather than
expose its richer :class:`Instrument` rows directly so existing
imports of :class:`SymbolEntry` / :func:`SymbolLookupService.search`
keep working without touching every caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import structlog

from dashboard.services.instruments import Instrument, InstrumentsService

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
    """Search the cached NSE instruments by partial symbol match.

    The service does NOT read ``config/universe.yaml`` any more; it
    delegates to :class:`InstrumentsService` so the same DuckDB rows
    drive the picker, the sweep form, and the legacy ``/api/symbols/*``
    endpoints.
    """

    def __init__(self, instruments: InstrumentsService) -> None:
        """Construct the service bound to a configured instruments cache."""
        self._instruments = instruments

    def list_symbols(self, *, limit: int = 50, q: str = "") -> list[SymbolEntry]:
        """Return up to ``limit`` ranked symbols matching ``q``.

        ``q`` defaults to the empty string which yields the first
        ``limit`` rows in alphabetical order. Behaviour mirrors
        :meth:`InstrumentsService.search`.
        """
        results = self._instruments.search(q, limit=limit)
        return [_to_entry(inst) for inst in results]

    def find_symbol(self, query: str) -> SymbolEntry | None:
        """Return the single entry whose tradingsymbol matches ``query``."""
        match = self._instruments.get_by_symbol(query)
        if match is None:
            return None
        return _to_entry(match)

    def search(self, query: str, *, limit: int = 20) -> list[SymbolEntry]:
        """Rank instruments against ``query`` and return up to ``limit`` matches."""
        return self.list_symbols(limit=limit, q=query)


def _to_entry(instrument: Instrument) -> SymbolEntry:
    token = instrument.instrument_token
    token_part = str(token) if token else "no token"
    return SymbolEntry(
        symbol=instrument.tradingsymbol,
        exchange=instrument.exchange,
        instrument_token=token if token else None,
        display_label=(
            f"{instrument.tradingsymbol} — {instrument.exchange} — {token_part}"
        ),
    )


__all__ = ["SymbolEntry", "SymbolLookupService"]
