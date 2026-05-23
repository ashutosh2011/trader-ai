"""Composite (combine) strategy — fuse N children into one trade decision.

The composite strategy wraps a list of registered :class:`Strategy`
instances and aggregates their per-bar :class:`Signal` outputs into a
single signal stream consumed by :class:`BacktestEngine`. The aggregation
is controlled by a small policy object:

* ``direction`` — how many children must agree on BUY/SELL for the
  composite to fire (``unanimous`` / ``majority`` / ``any``).
* ``price`` — how children's stop and target levels are combined
  (``tightest`` / ``widest`` / ``average``).

The composite intentionally lives in ``dashboard/services/`` (not in
``strategies/``) and is **not** registered via
:func:`register_strategy` — the dashboard wires it up directly when a
combine-mode run is requested, and the rest of the codebase never sees
it. Keeping it out of the registry preserves the schema-drift test in
``tests/test_dashboard_strategy_schemas.py``.

TRADEOFF: The :class:`Signal` model in this codebase only supports
``side="BUY"`` and ``side="SELL"`` — there is no ``"exit"`` action.
Children therefore never emit exits and the engine drives exits via
stop-loss / target hits on the open position. The original combine
spec spoke about an "exit propagation" rule; in this codebase the
equivalent guarantee is that a child abstention contributes zero
votes — open positions continue to exit on their per-trade SL/target
exactly as they would under the compare-mode flow.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import ClassVar, Literal

from core.context import Context
from core.signal import Signal
from indicators.base import Indicator
from strategies.base import Strategy

DirectionPolicy = Literal["unanimous", "majority", "any"]
PricePolicy = Literal["tightest", "widest", "average"]


@dataclass(frozen=True)
class CombinePolicy:
    """Policy controlling how child signals fuse into one composite signal.

    Attributes:
        direction: How many children must agree on the direction.

            * ``unanimous`` — every child must vote AND vote the same
              side. Abstentions break unanimity (an abstaining child is
              treated as a veto under this rule).
            * ``majority`` — strictly more votes for one side than the
              other AND zero votes for the opposite side. Opposing
              votes always veto; ties never fire.
            * ``any`` — at least one child fires AND no other child
              voted the opposite side. Conflicts (both sides) veto.

        price: How children's stop and target levels combine for the
            aggregated signal.

            * ``tightest`` — closest-to-entry SL and target. Safest
              levels, quickest exits.
            * ``widest`` — furthest-from-entry SL and target. Patient.
            * ``average`` — mean of children's SLs and targets.
    """

    direction: DirectionPolicy = "unanimous"
    price: PricePolicy = "tightest"


@dataclass(frozen=True)
class _ChildVote:
    """A single child's first signal and its display tag for this bar."""

    tag: str
    signal: Signal


class CompositeStrategy(Strategy):
    """Aggregate per-bar signals from N child strategies into one stream.

    The composite calls each child's :meth:`Strategy.on_bar` in order
    every bar, collects the first signal from each child (children may
    return multiple signals, but :class:`backtest.engine.BacktestEngine`
    only consumes the first; mirroring that keeps the vote consistent
    with the executed behaviour), and applies the :class:`CombinePolicy`:

    1.  Group child votes by ``side`` (BUY / SELL). Children that
        returned no signal contribute zero votes.
    2.  Apply :attr:`CombinePolicy.direction` to decide whether to fire
        and which side wins (no fire returns ``[]``).
    3.  Aggregate stop-loss and target across the winning side per
        :attr:`CombinePolicy.price`. Long aggregation maximises SL and
        minimises target for ``tightest``; short aggregation flips.
    4.  ``confidence`` is the mean of confidences from children that
        voted the winning side.
    5.  ``entry`` is the entry from the highest-confidence winning
        child. Every example strategy uses the bar close, so the
        picked value is also the bar close in normal use, but picking
        a single child's value keeps the composite numerically
        identical to that child when a single child wins under the
        ``any`` policy.
    6.  ``reasons`` are concatenated with a ``"<child_tag>: <reason>"``
        prefix; ``indicator_snapshot`` becomes a flat dict keyed by
        ``"<child_tag>.<key>"`` so the dashboard can display per-child
        contributions on the detail page.

    Child tags are :attr:`Strategy.id` by default. If two children
    share the same id (e.g. two ``ema_crossover`` instances with
    different params) the tags are deterministically suffixed with
    their 1-based occurrence index — ``ema_crossover_1``,
    ``ema_crossover_2`` — to keep the reason prefix and snapshot keys
    unambiguous.

    The composite is **not** in the strategy registry; construct it
    directly via :class:`BacktestRunner.run_combined`.
    """

    id: ClassVar[str] = "composite"
    timeframe: ClassVar[str] = "1m"

    def __init__(
        self,
        *,
        children: Sequence[Strategy],
        policy: CombinePolicy,
        symbol: str = "SYNTH",
    ) -> None:
        super().__init__()
        if len(children) < 2:
            msg = "CompositeStrategy requires at least 2 children"
            raise ValueError(msg)
        self._children: tuple[Strategy, ...] = tuple(children)
        self._policy = policy
        self._symbol = symbol
        self._tags: tuple[str, ...] = _build_child_tags(self._children)
        # Composite has no indicators of its own; each child manages
        # its own indicator state via the ``Strategy`` base class. We
        # advertise the union of child indicators so any caller poking
        # at ``required_indicators`` (e.g. warmup checks) sees the
        # full set, but we never call ``precompute_indicators`` on the
        # composite itself.
        merged: list[Indicator] = []
        for child in self._children:
            merged.extend(getattr(child, "required_indicators", []) or [])
        self.required_indicators = merged

    @property
    def policy(self) -> CombinePolicy:
        """Return the combine policy in effect."""
        return self._policy

    @property
    def children(self) -> tuple[Strategy, ...]:
        """Return the child strategy instances in selection order."""
        return self._children

    @property
    def child_tags(self) -> tuple[str, ...]:
        """Return per-child display tags (id with collision suffixes)."""
        return self._tags

    def on_bar(self, ctx: Context) -> list[Signal]:
        """Collect votes, apply policy, emit at most one composite signal."""
        votes: list[_ChildVote] = []
        for child, tag in zip(self._children, self._tags, strict=True):
            child_signals = child.on_bar(ctx)
            if not child_signals:
                continue
            votes.append(_ChildVote(tag=tag, signal=child_signals[0]))

        winning_side = resolve_direction(
            votes, total_children=len(self._children), policy=self._policy.direction
        )
        if winning_side is None:
            return []

        winners = [v for v in votes if v.signal.side == winning_side]
        # Pick max-confidence winner for ``entry`` to keep the composite
        # numerically aligned with the strongest contributing child.
        anchor = max(winners, key=lambda v: v.signal.confidence)
        entry = anchor.signal.entry
        stop_loss = aggregate_stop(
            winners=winners, side=winning_side, policy=self._policy.price
        )
        target = aggregate_target(
            winners=winners, side=winning_side, policy=self._policy.price
        )
        if not _direction_consistent(
            side=winning_side, entry=entry, stop_loss=stop_loss, target=target
        ):
            # Aggregation produced a stop/target that crossed the entry —
            # e.g. ``average`` of two long stops where one is unusually
            # wide. Skip cleanly rather than raise a Pydantic error
            # inside the engine (which swallows it as a silent no-op).
            return []

        confidence = float(fmean(v.signal.confidence for v in winners))
        reasons: list[str] = []
        indicator_snapshot: dict[str, float] = {}
        for v in winners:
            reasons.extend(f"{v.tag}: {reason}" for reason in v.signal.reasons)
            for key, value in v.signal.indicator_snapshot.items():
                indicator_snapshot[f"{v.tag}.{key}"] = value

        return [
            Signal(
                symbol=self._symbol,
                side=winning_side,
                entry=entry,
                stop_loss=stop_loss,
                target=target,
                qty=None,
                timeframe=self.timeframe,
                strategy_id=self.id,
                reasons=reasons,
                indicator_snapshot=indicator_snapshot,
                confidence=max(0.0, min(1.0, confidence)),
                ts=anchor.signal.ts,
            )
        ]


def _build_child_tags(children: Sequence[Strategy]) -> tuple[str, ...]:
    """Return display tags, suffixing duplicated ids deterministically."""
    counts = Counter(child.id for child in children)
    seen: dict[str, int] = {}
    tags: list[str] = []
    for child in children:
        if counts[child.id] == 1:
            tags.append(child.id)
            continue
        seen[child.id] = seen.get(child.id, 0) + 1
        tags.append(f"{child.id}_{seen[child.id]}")
    return tuple(tags)


def resolve_direction(
    votes: Sequence[_ChildVote],
    *,
    total_children: int,
    policy: DirectionPolicy,
) -> Literal["BUY", "SELL"] | None:
    """Apply the direction policy and return the winning side, or ``None``.

    ``votes`` only includes children that returned a signal this bar;
    abstentions are absent. ``total_children`` lets the ``unanimous``
    policy detect abstentions as vetoes.
    """
    if not votes:
        return None
    buys = sum(1 for v in votes if v.signal.side == "BUY")
    sells = sum(1 for v in votes if v.signal.side == "SELL")
    if policy == "unanimous":
        if buys == total_children and sells == 0:
            return "BUY"
        if sells == total_children and buys == 0:
            return "SELL"
        return None
    if policy == "majority":
        if buys > sells and sells == 0:
            return "BUY"
        if sells > buys and buys == 0:
            return "SELL"
        return None
    if policy == "any":
        if buys >= 1 and sells == 0:
            return "BUY"
        if sells >= 1 and buys == 0:
            return "SELL"
        return None
    return None  # pragma: no cover - exhaustive Literal


def aggregate_stop(
    *,
    winners: Sequence[_ChildVote],
    side: Literal["BUY", "SELL"],
    policy: PricePolicy,
) -> float:
    """Aggregate stop-loss levels across winning children per policy.

    Long ``tightest`` keeps the highest stop (closest below entry);
    long ``widest`` keeps the lowest stop. Short flips the comparison
    because a short's stop sits above the entry.
    """
    stops = [v.signal.stop_loss for v in winners]
    if policy == "average":
        return float(fmean(stops))
    if side == "BUY":
        return max(stops) if policy == "tightest" else min(stops)
    return min(stops) if policy == "tightest" else max(stops)


def aggregate_target(
    *,
    winners: Sequence[_ChildVote],
    side: Literal["BUY", "SELL"],
    policy: PricePolicy,
) -> float:
    """Aggregate target levels across winning children per policy."""
    targets = [v.signal.target for v in winners]
    if policy == "average":
        return float(fmean(targets))
    if side == "BUY":
        return min(targets) if policy == "tightest" else max(targets)
    return max(targets) if policy == "tightest" else min(targets)


def _direction_consistent(
    *,
    side: Literal["BUY", "SELL"],
    entry: float,
    stop_loss: float,
    target: float,
) -> bool:
    """Return True when SL/target satisfy the :class:`Signal` validator.

    Aggregation across children can occasionally collapse the
    direction inequalities (e.g. ``average`` of three longs where one
    had drifted to an unusually wide stop). Detecting it here lets the
    composite abstain instead of raising inside the engine.
    """
    if entry <= 0 or stop_loss <= 0 or target <= 0:
        return False
    if side == "BUY":
        return stop_loss < entry < target
    return target < entry < stop_loss


__all__ = [
    "CombinePolicy",
    "CompositeStrategy",
    "DirectionPolicy",
    "PricePolicy",
    "aggregate_stop",
    "aggregate_target",
    "resolve_direction",
]
