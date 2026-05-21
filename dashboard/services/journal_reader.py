"""Read JSONL journal entries with simple filter + pagination.

The journal is an append-only file written by :class:`journal.log.TradingJournal`.
We read it forward in O(file size) per call — adequate for personal-use
volumes (low thousands of lines per day). For larger journals we'd switch
to a DuckDB-backed view, but the JSONL file is canonical so reading it
directly avoids a sync layer.

TRADEOFF: We re-read the whole file on every tail call rather than
maintaining a file-offset cursor. The cost is O(n) per refresh; the
benefit is that the reader is stateless and survives log-rotation /
truncation without crashing the dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class JournalEntry:
    """One parsed journal line."""

    ts: str
    event: str
    payload: dict[str, Any]
    raw: str

    @property
    def symbol(self) -> str | None:
        """Extract the symbol from common event payload shapes (or ``None``)."""
        if "symbol" in self.payload and isinstance(self.payload["symbol"], str):
            return self.payload["symbol"]
        for key in ("signal", "verdict", "order"):
            nested = self.payload.get(key)
            if isinstance(nested, dict):
                sym = nested.get("symbol")
                if isinstance(sym, str):
                    return sym
        return None


@dataclass(frozen=True)
class TailResult:
    """Page of journal entries plus the latest seen timestamp."""

    entries: list[JournalEntry]
    last_ts: str | None


class JournalReader:
    """Tail and filter a JSONL trading journal file."""

    def __init__(self, path: Path | None) -> None:
        """Construct a reader for ``path``.

        Args:
            path: Path to the JSONL journal. ``None`` produces a reader
                that returns empty results — used when no journal is
                configured.
        """
        self._path = path

    @property
    def path(self) -> Path | None:
        """Path to the journal file being read."""
        return self._path

    def exists(self) -> bool:
        """Return whether the journal file is currently present."""
        return self._path is not None and self._path.is_file()

    def tail(
        self,
        *,
        limit: int = 100,
        since_ts: str | None = None,
        event_types: list[str] | None = None,
        symbol: str | None = None,
    ) -> TailResult:
        """Return up to ``limit`` most-recent matching entries.

        Args:
            limit: Maximum number of entries to return.
            since_ts: Only include entries with ``ts > since_ts``.
            event_types: Filter to these event types (exact match).
            symbol: Filter to entries that resolve to this symbol.

        Returns:
            A :class:`TailResult` with the most-recent entries first and
            ``last_ts`` set to the newest timestamp seen overall (used by
            the UI to drive incremental ``since_ts`` polling).
        """
        if not self.exists():
            return TailResult(entries=[], last_ts=since_ts)

        assert self._path is not None  # narrowed by exists()
        entries: list[JournalEntry] = []
        last_seen: str | None = since_ts
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "dashboard_journal_bad_line",
                            preview=line[:120],
                        )
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    ts = parsed.get("ts")
                    event = parsed.get("event")
                    if not isinstance(ts, str) or not isinstance(event, str):
                        continue
                    if last_seen is None or ts > last_seen:
                        last_seen = ts
                    if since_ts is not None and ts <= since_ts:
                        continue
                    if event_types and event not in event_types:
                        continue
                    payload = {k: v for k, v in parsed.items() if k not in {"ts", "event"}}
                    entry = JournalEntry(ts=ts, event=event, payload=payload, raw=line)
                    if symbol is not None and entry.symbol != symbol:
                        continue
                    entries.append(entry)
        except OSError as exc:
            logger.warning("dashboard_journal_read_failed", error=str(exc))
            return TailResult(entries=[], last_ts=since_ts)

        entries.reverse()
        return TailResult(entries=entries[:limit], last_ts=last_seen)

    def today_realized_pnl(self) -> float:
        """Sum realized PnL across order events whose ts falls in the current day.

        We look at ``order`` events whose payload includes a numeric
        ``pnl`` (paper broker close events) — these are the only journal
        rows that carry realized PnL today. Returns ``0.0`` when no
        journal is configured.
        """
        if not self.exists():
            return 0.0
        today = datetime.now().astimezone().date().isoformat()
        total = 0.0
        assert self._path is not None
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    ts = parsed.get("ts")
                    if not isinstance(ts, str) or not ts.startswith(today):
                        continue
                    pnl = _extract_pnl(parsed)
                    if pnl is not None:
                        total += pnl
        except OSError:
            return 0.0
        return total

    def event_counts_today(self) -> dict[str, int]:
        """Return ``{event_type: count}`` for the current day."""
        if not self.exists():
            return {}
        counts: dict[str, int] = {}
        today = datetime.now().astimezone().date().isoformat()
        assert self._path is not None
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    ts = parsed.get("ts")
                    event = parsed.get("event")
                    if not isinstance(ts, str) or not isinstance(event, str):
                        continue
                    if not ts.startswith(today):
                        continue
                    counts[event] = counts.get(event, 0) + 1
        except OSError:
            return {}
        return counts


def _extract_pnl(parsed: dict[str, Any]) -> float | None:
    """Pull a numeric PnL out of an order event payload, if present."""
    for key in ("pnl", "realized_pnl"):
        value = parsed.get(key)
        if isinstance(value, int | float):
            return float(value)
    order = parsed.get("order")
    if isinstance(order, dict):
        for key in ("pnl", "realized_pnl"):
            value = order.get(key)
            if isinstance(value, int | float):
                return float(value)
    return None
