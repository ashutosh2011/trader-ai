"""Co-decide orchestration loop: strategy → risk → analyst → broker."""

from dataclasses import dataclass, field

import pandas as pd
import structlog

from analyst.analyst import Analyst
from config.settings import AppSettings
from core.context import Context
from core.instrument import Instrument
from core.signal import Signal
from data.feed import BarFeed
from execution.broker import Broker
from execution.kite import KiteBroker
from execution.order_state import OrderStateStore
from execution.paper import PaperBroker
from journal.log import TradingJournal
from orchestrator.scheduler import MarketScheduler
from risk.manager import RiskManager, RiskState

logger = structlog.get_logger(__name__)


@dataclass
class LoopStats:
    """Counters from a paper/replay loop run."""

    signals_seen: int = 0
    risk_rejected: int = 0
    analyst_vetoed: int = 0
    orders_placed: int = 0
    bar_exits: int = 0


@dataclass
class LoopResult:
    """Outcome of running the orchestrator loop."""

    stats: LoopStats
    realized_pnl: float
    open_positions: int


@dataclass
class OrchestratorLoop:
    """Wire strategy signals through risk, optional analyst, and broker.

    Attributes:
        override_qty: Optional hard ceiling on order quantity applied AFTER
            risk sizing AND ``post_check``. ``None`` (default) keeps the
            risk-derived qty. ``0`` causes every signal to be rejected.
            TRADEOFF: the cap is re-floored to lot size after capping so the
            final qty stays exchange-compliant.
    """

    strategy: object
    broker: Broker
    risk: RiskManager
    feed: BarFeed
    journal: TradingJournal | None = None
    analyst: Analyst | None = None
    settings: AppSettings = field(default_factory=AppSettings)
    scheduler: MarketScheduler | None = None
    instrument: Instrument | None = None
    override_qty: int | None = None
    state_store: OrderStateStore | None = None

    async def run(self) -> LoopResult:
        """Process all bars from the feed."""
        frame = self.feed.to_dataframe()
        # TRADEOFF: the orchestrator runs one bar stream at a time; the bar's
        # "symbol" column (if present) is informational only. Position lookups
        # use the strategy/signal symbol so user-overridden symbols (e.g.
        # --symbol RELIANCE on synthetic SYNTH bars) still exit correctly.
        # Multi-symbol bar streams will need per-symbol bar feeds; that's a
        # Batch 4+ concern.
        strategy_symbol = self._strategy_symbol(frame)
        stats = LoopStats()
        state = RiskState(
            account_equity=self._account_equity(),
            open_positions=0,
        )
        last_prices: dict[str, float] = {}
        for bar_index in range(len(frame)):
            row = frame.iloc[bar_index]
            ts = pd.Timestamp(row["timestamp"]).to_pydatetime()

            if self.scheduler is not None:
                schedule = self.scheduler.should_process_bar(
                    ts, instrument_type=state.instrument_type
                )
                if not schedule.allowed:
                    continue

            # Drive the GTT-based broker's polling state machine before any
            # exit / entry logic so the loop reacts to fills + GTT triggers
            # that landed since the previous tick.
            if isinstance(self.broker, KiteBroker):
                self.broker.poll_and_advance()

            if isinstance(self.broker, PaperBroker):
                high = float(row["high"])
                low = float(row["low"])
                close = float(row["close"])
                # Iterate open positions; the bar's high/low is applied to
                # every position since we have one active symbol per loop.
                for position in self.broker.get_positions():
                    exit_price, reason = self.broker.check_bar_exits(
                        position.symbol, high, low
                    )
                    if exit_price is not None:
                        pnl = self.broker.close_position(
                            position.symbol, exit_price, reason
                        )
                        state.daily_realized_pnl += pnl
                        state.open_positions = len(self.broker.get_positions())
                        stats.bar_exits += 1
                    last_prices[position.symbol] = close
                last_prices[strategy_symbol] = close
                state.daily_unrealized_pnl = self.broker.mark_to_market(last_prices)

            ctx = Context(
                symbol=strategy_symbol,
                bars=frame,
                bar_index=bar_index,
                timestamp=ts,
                timeframe=getattr(self.strategy, "timeframe", "1m"),
            )
            signals: list[Signal] = self.strategy.on_bar(ctx)  # type: ignore[attr-defined]
            if not signals:
                continue

            for signal in signals[:1]:
                stats.signals_seen += 1
                await self._process_signal(signal, ctx, state, stats)

            state.open_positions = len(self.broker.get_positions())

        realized = 0.0
        if isinstance(self.broker, PaperBroker):
            realized = self.broker.account.realized_pnl
        return LoopResult(
            stats=stats,
            realized_pnl=realized,
            open_positions=len(self.broker.get_positions()),
        )

    async def _process_signal(
        self,
        signal: Signal,
        ctx: Context,
        state: RiskState,
        stats: LoopStats,
    ) -> None:
        decision = self.risk.pre_check(signal, ctx, state, instrument=self.instrument)
        if self.journal:
            self.journal.write_signal(signal)
            self.journal.write_risk_decision(decision, signal.symbol)
        if not decision.approved:
            stats.risk_rejected += 1
            logger.info("signal_rejected", reason=decision.reason, symbol=signal.symbol)
            return

        size_multiplier = 1.0
        if self.analyst is not None:
            verdict = await self.analyst.analyze(signal, ctx)
            if self.journal:
                self.journal.write_verdict(verdict, signal.symbol)
            if verdict.action == "VETO":
                stats.analyst_vetoed += 1
                logger.info("signal_vetoed", symbol=signal.symbol, rationale=verdict.rationale)
                return
            size_multiplier = verdict.size_multiplier

        qty = self.risk.size(
            signal,
            ctx,
            state,
            instrument=self.instrument,
            size_multiplier=size_multiplier,
        )
        post = self.risk.post_check(signal, ctx, state, qty, instrument=self.instrument)
        if not post.approved:
            stats.risk_rejected += 1
            logger.info("signal_post_rejected", reason=post.reason, symbol=signal.symbol)
            return

        capped_qty = self._apply_qty_cap(qty)
        if capped_qty <= 0:
            stats.risk_rejected += 1
            logger.info(
                "signal_qty_cap_zero",
                symbol=signal.symbol,
                risk_qty=qty,
                override_qty=self.override_qty,
            )
            return
        if capped_qty != qty:
            logger.info(
                "signal_qty_capped",
                symbol=signal.symbol,
                risk_qty=qty,
                override_qty=self.override_qty,
                final_qty=capped_qty,
            )

        order = self.broker.place_bracket_order(signal, capped_qty)
        if self.journal:
            self.journal.write_order(order)
        if order.status == "FILLED":
            stats.orders_placed += 1
            state.open_positions = len(self.broker.get_positions())

    def _account_equity(self) -> float:
        if isinstance(self.broker, PaperBroker):
            return self.broker.account.equity
        return self.settings.paper.account_equity

    def _strategy_symbol(self, frame: pd.DataFrame) -> str:
        """Resolve the active symbol for the loop.

        Prefers the strategy's own configured symbol (e.g. ``--symbol RELIANCE``)
        over the bar frame's symbol column, so user overrides flow through to
        position lookups in the broker.
        """
        for attr in ("_symbol", "symbol"):
            value = getattr(self.strategy, attr, None)
            if isinstance(value, str) and value:
                return value
        if "symbol" in frame.columns and len(frame) > 0:
            return str(frame["symbol"].iloc[0])
        return "UNKNOWN"

    def _apply_qty_cap(self, qty: int) -> int:
        """Apply ``override_qty`` cap and re-floor to lot size.

        TRADEOFF: capping after risk-sizing may produce a non-lot-aligned
        value; we re-floor to the instrument lot size (defaulting to 1)
        so the final qty respects exchange constraints. Callers that pass
        ``override_qty=0`` will get back ``0`` and the signal is rejected.
        """
        if self.override_qty is None:
            return qty
        if self.override_qty <= 0:
            return 0
        lot_size = self.instrument.lot_size if self.instrument is not None else 1
        capped = min(qty, self.override_qty)
        return (capped // lot_size) * lot_size
