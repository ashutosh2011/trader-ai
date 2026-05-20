"""A/B test: rules-only vs co-decide on the same replay feed."""

from dataclasses import dataclass

import structlog

from analyst.analyst import Analyst
from analyst.provider import LLMProvider
from config.settings import AppSettings
from data.feed import BarFeed
from execution.paper import PaperBroker
from orchestrator.loop import OrchestratorLoop
from risk.manager import RiskManager
from strategies.base import Strategy

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AbTestArmResult:
    """Results for one A/B arm."""

    label: str
    signals_seen: int
    orders_placed: int
    risk_rejected: int
    analyst_vetoed: int
    realized_pnl: float
    bar_exits: int


@dataclass(frozen=True)
class AbTestResult:
    """Comparison of rules-only vs co-decide."""

    rules_only: AbTestArmResult
    co_decide: AbTestArmResult

    @property
    def veto_count(self) -> int:
        return self.co_decide.analyst_vetoed


async def run_ab_test(
    strategy: Strategy,
    feed: BarFeed,
    *,
    analyst_provider: LLMProvider,
    settings: AppSettings | None = None,
) -> AbTestResult:
    """Run the same replay feed twice: without and with analyst."""
    app = settings or AppSettings()

    rules_loop = _build_loop(strategy, feed, app, analyst=None)
    rules_result = await rules_loop.run()

    co_loop = _build_loop(
        strategy,
        feed,
        app,
        analyst=Analyst(analyst_provider),
    )
    co_result = await co_loop.run()

    return AbTestResult(
        rules_only=AbTestArmResult(
            label="rules_only",
            signals_seen=rules_result.stats.signals_seen,
            orders_placed=rules_result.stats.orders_placed,
            risk_rejected=rules_result.stats.risk_rejected,
            analyst_vetoed=rules_result.stats.analyst_vetoed,
            realized_pnl=rules_result.realized_pnl,
            bar_exits=rules_result.stats.bar_exits,
        ),
        co_decide=AbTestArmResult(
            label="co_decide",
            signals_seen=co_result.stats.signals_seen,
            orders_placed=co_result.stats.orders_placed,
            risk_rejected=co_result.stats.risk_rejected,
            analyst_vetoed=co_result.stats.analyst_vetoed,
            realized_pnl=co_result.realized_pnl,
            bar_exits=co_result.stats.bar_exits,
        ),
    )


def _build_loop(
    strategy: Strategy,
    feed: BarFeed,
    settings: AppSettings,
    *,
    analyst: Analyst | None,
) -> OrchestratorLoop:
    broker = PaperBroker(settings=settings)
    return OrchestratorLoop(
        strategy=strategy,
        broker=broker,
        risk=RiskManager(settings),
        feed=feed,
        analyst=analyst,
        settings=settings,
    )
