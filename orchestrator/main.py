"""Click CLI for backtest, paper trading, live, and A/B tests."""

import asyncio
from pathlib import Path

import click
import structlog

from analyst.providers.mock import MockLLMProvider
from backtest.engine import BacktestEngine
from backtest.metrics import compute_performance_metrics
from backtest.report import write_html_report, write_markdown_report
from config.settings import load_settings
from data.kite_client import KiteClient
from data.replay_feed import ReplayFeed
from data.synthetic import make_synthetic_bars
from execution.kite import KiteBroker
from execution.paper import PaperBroker
from execution.reconciler import StateReconciler
from journal.log import TradingJournal
from orchestrator.ab_test import run_ab_test
from orchestrator.kite_login import run_interactive_login
from orchestrator.loop import OrchestratorLoop
from orchestrator.scheduler import MarketScheduler
from risk.manager import RiskManager, is_kill_switch_active
from strategies.examples.ema_crossover import EmaCrossover

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger(__name__)


@click.group()
def cli() -> None:
    """Tradebot orchestrator CLI."""


@cli.command("backtest")
@click.option("--bars", type=click.Path(exists=True), default=None)
@click.option("--bars-count", default=1000, show_default=True)
@click.option("--qty", default=1, show_default=True)
@click.option("--symbol", default="SYNTH", show_default=True)
@click.option("--report-md", type=click.Path(), default=None)
@click.option("--report-html", type=click.Path(), default=None)
def backtest_cmd(
    bars: str | None,
    bars_count: int,
    qty: int,
    symbol: str,
    report_md: str | None,
    report_html: str | None,
) -> None:
    """Run backtest, print metrics, optionally write reports."""
    frame = (
        ReplayFeed(Path(bars)).to_dataframe()
        if bars is not None
        else make_synthetic_bars(bars_count, seed=42)
    )
    strategy = EmaCrossover(symbol=symbol)
    result = BacktestEngine(qty=qty).run(strategy, frame)
    metrics = compute_performance_metrics(result)
    click.echo(f"trades={metrics.total_trades} total_pnl={metrics.total_pnl:.2f}")
    click.echo(f"sharpe={metrics.sharpe_ratio:.4f} mdd_pct={metrics.max_drawdown_pct:.2f}%")
    click.echo(f"win_rate={metrics.win_rate_pct:.2f}% profit_factor={metrics.profit_factor:.4f}")
    if report_md:
        path = write_markdown_report(Path(report_md), result, metrics=metrics)
        click.echo(f"wrote markdown report: {path}")
    if report_html:
        path = write_html_report(Path(report_html), result, metrics=metrics)
        click.echo(f"wrote html report: {path}")


@cli.command("paper")
@click.option("--bars", type=click.Path(exists=True), default=None)
@click.option("--bars-count", default=500, show_default=True)
@click.option("--symbol", default="SYNTH", show_default=True)
@click.option("--journal", type=click.Path(), default=None)
@click.option("--config", type=click.Path(), default=None)
@click.option("--schedule/--no-schedule", default=False, help="Apply NSE market-hours filter")
def paper_cmd(
    bars: str | None,
    bars_count: int,
    symbol: str,
    journal: str | None,
    config: str | None,
    schedule: bool,
) -> None:
    """Run paper loop on replay feed (rules-only, no analyst)."""
    settings = load_settings(Path(config) if config else None)
    frame = (
        ReplayFeed(Path(bars)).to_dataframe()
        if bars is not None
        else make_synthetic_bars(bars_count, seed=42)
    )
    feed = ReplayFeed(frame)
    strategy = EmaCrossover(symbol=symbol)
    loop = OrchestratorLoop(
        strategy=strategy,
        broker=PaperBroker(settings=settings),
        risk=RiskManager(settings),
        feed=feed,
        journal=TradingJournal(Path(journal)) if journal else None,
        settings=settings,
        scheduler=MarketScheduler(settings) if schedule else None,
    )
    result = asyncio.run(loop.run())
    click.echo(
        f"signals={result.stats.signals_seen} orders={result.stats.orders_placed} "
        f"pnl={result.realized_pnl:.2f} exits={result.stats.bar_exits}"
    )


@cli.command("live")
@click.option("--bars", type=click.Path(exists=True), default=None)
@click.option("--bars-count", default=200, show_default=True)
@click.option("--symbol", default="SYNTH", show_default=True)
@click.option(
    "--qty",
    default=None,
    type=int,
    help="Order qty (default: settings.live_default_qty)",
)
@click.option("--journal", type=click.Path(), default=None)
@click.option("--config", type=click.Path(), default=None)
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    help="Use replay feed when Kite credentials missing (default: dry-run)",
)
def live_cmd(
    bars: str | None,
    bars_count: int,
    symbol: str,
    qty: int | None,
    journal: str | None,
    config: str | None,
    dry_run: bool,
) -> None:
    """Run live loop with kill switch armed and minimal size.

    WARNING: Real orders when Kite credentials are configured and --no-dry-run.
    """
    settings = load_settings(Path(config) if config else None)
    if is_kill_switch_active(
        kill_file=settings.kill_switch_file,
        env_var=settings.kill_switch_env,
    ):
        click.echo("KILL switch active — aborting live run")
        raise SystemExit(1)

    order_qty = qty if qty is not None else settings.live_default_qty
    frame = (
        ReplayFeed(Path(bars)).to_dataframe()
        if bars is not None
        else make_synthetic_bars(bars_count, seed=42)
    )
    feed = ReplayFeed(frame)

    broker: PaperBroker | KiteBroker
    if settings.kite_configured() and not dry_run:
        client = KiteClient.from_settings(settings)
        reconciler = StateReconciler(KiteBroker(client, settings=settings))
        state = reconciler.reconcile(orders=client.orders())
        click.echo(
            f"reconciled positions={len(state.positions)} "
            f"open_orders={len(state.open_orders)} drift={state.drift_symbols}"
        )
        broker = KiteBroker(client, settings=settings)
        click.echo("LIVE MODE: real Kite orders enabled")
    else:
        broker = PaperBroker(settings=settings)
        if not settings.kite_configured():
            click.echo("Kite credentials missing — using paper broker (dry-run)")
        else:
            click.echo("dry-run: paper broker on replay feed")

    strategy = EmaCrossover(symbol=symbol)
    loop = OrchestratorLoop(
        strategy=strategy,
        broker=broker,
        risk=RiskManager(settings),
        feed=feed,
        journal=TradingJournal(Path(journal)) if journal else None,
        settings=settings,
        scheduler=MarketScheduler(settings),
    )
    result = asyncio.run(loop.run())
    click.echo(
        f"live signals={result.stats.signals_seen} orders={result.stats.orders_placed} "
        f"qty_cap={order_qty} open={result.open_positions}"
    )


@cli.command("kite-login")
@click.option(
    "--request-token",
    default=None,
    help="One-time token from Kite redirect (else prompted)",
)
@click.option("--config", type=click.Path(), default=None)
@click.option(
    "--no-save",
    is_flag=True,
    help="Print token only; do not write KITE_ACCESS_TOKEN to .env",
)
def kite_login_cmd(
    request_token: str | None,
    config: str | None,
    no_save: bool,
) -> None:
    """Obtain a daily Kite access token via browser login."""
    settings = load_settings(Path(config) if config else None)
    if not settings.kite.api_key or not settings.kite.api_secret:
        click.echo("Set KITE_API_KEY and KITE_API_SECRET in .env first.")
        raise SystemExit(1)
    run_interactive_login(
        settings.kite.api_key,
        settings.kite.api_secret,
        request_token=request_token,
        save=not no_save,
    )


@cli.command("ab-test")
@click.option("--bars-count", default=500, show_default=True)
@click.option("--symbol", default="SYNTH", show_default=True)
@click.option("--veto/--no-veto", default=False, help="Mock analyst vetoes all signals")
def ab_test_cmd(bars_count: int, symbol: str, veto: bool) -> None:
    """Compare rules-only vs co-decide on the same replay feed."""
    frame = make_synthetic_bars(bars_count, seed=42)
    feed = ReplayFeed(frame)
    strategy = EmaCrossover(symbol=symbol)
    if veto:
        response = (
            '{"action": "VETO", "size_multiplier": 0.0, '
            '"confidence": 0.9, "rationale": "test veto"}'
        )
    else:
        response = (
            '{"action": "APPROVE", "size_multiplier": 0.8, '
            '"confidence": 0.8, "rationale": "test approve"}'
        )
    provider = MockLLMProvider(response)
    result = asyncio.run(run_ab_test(strategy, feed, analyst_provider=provider))
    click.echo("=== Rules Only ===")
    click.echo(
        f"signals={result.rules_only.signals_seen} orders={result.rules_only.orders_placed} "
        f"pnl={result.rules_only.realized_pnl:.2f}"
    )
    click.echo("=== Co-Decide ===")
    click.echo(
        f"signals={result.co_decide.signals_seen} orders={result.co_decide.orders_placed} "
        f"vetoed={result.co_decide.analyst_vetoed} pnl={result.co_decide.realized_pnl:.2f}"
    )


if __name__ == "__main__":
    cli()
