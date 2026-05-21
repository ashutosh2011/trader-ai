"""Read + admin-mark orders from the persistent :class:`OrderStateStore`.

The orders page filters / paginates ``OrderRecord`` rows and lets an
operator manually flip a stuck row into a terminal state (FAILED /
CANCELLED). Marking is the dashboard's only write into the order store —
all other writes are made by the broker — and we keep it deliberately
narrow: only OPEN → terminal transitions are allowed, and we always
stamp ``updated_at`` and the operator-source error message.

TRADEOFF: We do **not** ask the broker to cancel anything when the
operator marks a row CANCELLED. The broker keeps its own GTT / order
state; marking is a *local* override for cases where the broker reports
something orphaned that the dashboard view can't otherwise clear. The
``/strategies`` and ``/live`` pages document this so users know to run
``flatten`` for the real broker-side cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import structlog

from execution.order_state import OPEN_STATES, OrderRecord, OrderState, OrderStateStore

logger = structlog.get_logger(__name__)

_TERMINAL_MARK_STATES: frozenset[OrderState] = frozenset(
    {OrderState.FAILED, OrderState.CANCELLED}
)


@dataclass(frozen=True)
class OrdersPage:
    """One page of order rows with pagination metadata."""

    rows: list[OrderRecord]
    page: int
    per_page: int
    total: int

    @property
    def total_pages(self) -> int:
        """Total pages given ``total`` rows at ``per_page`` per page."""
        if self.per_page <= 0:
            return 1
        return max(1, (self.total + self.per_page - 1) // self.per_page)


class OrdersService:
    """Filter, paginate, and admin-mark :class:`OrderRecord` rows."""

    def __init__(self, store: OrderStateStore) -> None:
        """Construct the service bound to a live :class:`OrderStateStore`."""
        self._store = store

    def list_open(self) -> list[OrderRecord]:
        """Return every record in an OPEN state (``PENDING_ENTRY``/``ENTERED``)."""
        return self._store.list_open()

    def list_all(self) -> list[OrderRecord]:
        """Return every record in the store, oldest first."""
        return self._store.list_all()

    def page(
        self,
        *,
        state: str = "ALL",
        symbol: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> OrdersPage:
        """Return a filtered + paginated page of records.

        Args:
            state: ``"ALL"`` or a :class:`OrderState` value (string form).
            symbol: Case-insensitive substring filter on the symbol.
            page: 1-indexed page number.
            per_page: Number of rows per page.

        Returns:
            An :class:`OrdersPage` (newest-first).
        """
        rows = sorted(
            self._store.list_all(),
            key=lambda record: record.created_at,
            reverse=True,
        )
        if state != "ALL":
            try:
                wanted = OrderState(state)
            except ValueError:
                wanted = None
            if wanted is not None:
                rows = [record for record in rows if record.state == wanted]
        if symbol:
            needle = symbol.upper()
            rows = [record for record in rows if needle in record.symbol.upper()]
        total = len(rows)
        page = max(1, page)
        per_page = max(1, per_page)
        start = (page - 1) * per_page
        end = start + per_page
        return OrdersPage(
            rows=rows[start:end],
            page=page,
            per_page=per_page,
            total=total,
        )

    def mark(
        self,
        client_order_id: str,
        *,
        state: OrderState,
        reason: str = "marked_by_dashboard",
    ) -> OrderRecord | None:
        """Force a record into ``FAILED`` or ``CANCELLED``.

        Args:
            client_order_id: Target record id.
            state: New terminal state — must be ``FAILED`` or ``CANCELLED``.
            reason: Short string written to the record's ``error`` column.

        Returns:
            The updated record, or ``None`` if no such record exists.

        Raises:
            ValueError: If ``state`` is not a permitted terminal state.
            RuntimeError: If the record is already in a terminal state.
        """
        if state not in _TERMINAL_MARK_STATES:
            msg = f"mark() requires FAILED or CANCELLED, got {state.value}"
            raise ValueError(msg)
        existing = self._store.get(client_order_id)
        if existing is None:
            return None
        if existing.state not in OPEN_STATES:
            msg = (
                f"refusing to mark non-open record (client_order_id={client_order_id}, "
                f"state={existing.state.value})"
            )
            raise RuntimeError(msg)
        updated = existing.model_copy(
            update={
                "state": state,
                "error": reason,
                "updated_at": datetime.now().astimezone(),
            }
        )
        self._store.upsert(updated)
        logger.warning(
            "dashboard_order_marked",
            client_order_id=client_order_id,
            state=state.value,
            reason=reason,
        )
        return updated
