"""Co-decide orchestration loop: strategy → risk → analyst → broker."""

from dataclasses import dataclass, field

import pandas as pd
import structlog

from analyst.analyst import Analyst
from config.settings import AppSettings
from core.context import Context
from core.signal import Signal
from data.feed import BarFeed
from execution.broker import Broker
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
    """Wire strategy signals through risk, optional analyst, and broker."""

    strategy: object
    broker: Broker
    risk: RiskManager
    feed: BarFeed
    journal: TradingJournal | None = None
    analyst: Analyst | None = None
    settings: AppSettings = field(default_factory=AppSettings)
    scheduler: MarketScheduler | None = None

    async def run(self) -> LoopResult:
        """Process all bars from the feed."""
        frame = self.feed.to_dataframe()
        symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "UNKNOWN"
        stats = LoopStats()
        state = RiskState(
            account_equity=self._account_equity(),
            open_positions=0,
        )
        for bar_index in range(len(frame)):
            row = frame.iloc[bar_index]
            ts = pd.Timestamp(row["timestamp"]).to_pydatetime()

            if self.scheduler is not None:
                schedule = self.scheduler.should_process_bar(
                    ts, instrument_type=state.instrument_type
                )
                if not schedule.allowed:
                    continue

            if isinstance(self.broker, PaperBroker):
                exit_price, reason = self.broker.check_bar_exits(
                    symbol, float(row["high"]), float(row["low"])
                )
                if exit_price is not None:
                    pnl = self.broker.close_position(symbol, exit_price, reason)
                    state.daily_realized_pnl += pnl
                    state.open_positions = len(self.broker.get_positions())
                    stats.bar_exits += 1

            ctx = Context(
                symbol=symbol,
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
        decision = self.risk.pre_check(signal, ctx, state)
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

        qty = self.risk.size(signal, ctx, state, size_multiplier=size_multiplier)
        post = self.risk.post_check(signal, ctx, state, qty)
        if not post.approved:
            stats.risk_rejected += 1
            logger.info("signal_post_rejected", reason=post.reason, symbol=signal.symbol)
            return

        order = self.broker.place_bracket_order(signal, qty)
        if self.journal:
            self.journal.write_order(order)
        if order.status == "FILLED":
            stats.orders_placed += 1
            state.open_positions = len(self.broker.get_positions())

    def _account_equity(self) -> float:
        if isinstance(self.broker, PaperBroker):
            return self.broker.account.equity
        return self.settings.paper.account_equity
