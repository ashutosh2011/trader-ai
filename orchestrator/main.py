"""Click CLI for backtest, paper trading, live, and A/B tests."""

import asyncio
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import click
import duckdb
import structlog

from analyst.providers.mock import MockLLMProvider
from backtest.engine import BacktestEngine
from backtest.metrics import compute_performance_metrics
from backtest.report import write_html_report, write_markdown_report
from config.settings import AppSettings, load_settings
from dashboard.services.screener_service import (
    PROVIDER_OPTIONS,
    ScreenerProviderName,
    ScreenerService,
)
from dashboard.services.tuner_service import (
    PROVIDER_OPTIONS as TUNER_PROVIDER_OPTIONS,
)
from dashboard.services.tuner_service import (
    TunerProviderName,
    render_run_json,
    render_run_table,
)
from dashboard.state import AppState
from data.kite_client import KiteClient
from data.live_feed import LiveKiteFeed
from data.replay_feed import ReplayFeed
from data.synthetic import make_synthetic_bars
from execution.broker import FlattenIncomplete
from execution.kite import KiteBroker
from execution.order_state import OrderStateStore
from execution.paper import PaperBroker
from execution.reconciler import StateReconciler
from journal.log import TradingJournal
from orchestrator.ab_test import run_ab_test
from orchestrator.kite_login import run_interactive_login
from orchestrator.loop import OrchestratorLoop
from orchestrator.scheduler import MarketScheduler
from risk.manager import RiskManager, is_kill_switch_active
from screener.prompt import MarketContext
from screener.runner import render_run_record_json, render_run_record_table
from screener.store import (
    SCREENER_PICKS_SCHEMA,
    SCREENER_RUNS_SCHEMA,
    ScreenerStore,
)
from screener.universe import load_universe
from strategies.examples.ema_crossover import EmaCrossover

IST = ZoneInfo("Asia/Kolkata")

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
    help="HARD CEILING on order qty applied after risk sizing (default: settings.live_default_qty)",
)
@click.option("--journal", type=click.Path(), default=None)
@click.option("--config", type=click.Path(), default=None)
@click.option(
    "--state-db",
    "state_db",
    type=click.Path(),
    default=None,
    help="Path to the persistent OrderStateStore DuckDB file "
    "(default: settings.state_db_path).",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    help="Use replay feed when Kite credentials missing (default: dry-run)",
)
@click.option(
    "--live-feed",
    "live_feed",
    type=click.Choice(["replay", "kite"]),
    default="replay",
    show_default=True,
    help="Bar source. 'replay' = CSV/synthetic; 'kite' = LiveKiteFeed (requires credentials).",
)
@click.option(
    "--allow-replay-live",
    is_flag=True,
    default=False,
    help=(
        "Required to run --no-dry-run against replay/CSV bars. "
        "DANGEROUS: real orders may be placed against stale data."
    ),
)
def live_cmd(
    bars: str | None,
    bars_count: int,
    symbol: str,
    qty: int | None,
    journal: str | None,
    config: str | None,
    state_db: str | None,
    dry_run: bool,
    live_feed: str,
    allow_replay_live: bool,
) -> None:
    """Run live loop with kill switch armed and minimal size.

    WARNING: Real orders when Kite credentials are configured and --no-dry-run.
    Safety rails for ``--no-dry-run``:

    * ``--live-feed kite`` is required unless ``--allow-replay-live`` is set.
    * Synthetic bars are refused outright in ``--no-dry-run`` mode.
    * Missing Kite credentials in ``--live-feed kite`` is a hard error.
    """
    settings = load_settings(Path(config) if config else None)
    if is_kill_switch_active(
        kill_file=settings.kill_switch_file,
        env_var=settings.kill_switch_env,
    ):
        click.echo("KILL switch active — aborting live run")
        raise SystemExit(1)

    order_qty = qty if qty is not None else settings.live_default_qty
    if order_qty == 0:
        click.echo("--qty 0 rejected: nothing to trade")
        raise SystemExit(1)

    feed = _build_feed(
        bars=bars,
        bars_count=bars_count,
        live_feed=live_feed,
        dry_run=dry_run,
        allow_replay_live=allow_replay_live,
        settings=settings,
    )

    state_db_path = Path(state_db) if state_db else settings.state_db_path
    store: OrderStateStore | None = None
    broker: PaperBroker | KiteBroker
    if settings.kite_configured() and not dry_run:
        store = OrderStateStore(state_db_path)
        _print_state_summary(store, state_db_path)
        client = KiteClient.from_settings(settings)
        kite_broker = KiteBroker(client, settings=settings, state_store=store)
        reconciler = StateReconciler(kite_broker, state_store=store)
        reconciled = reconciler.reconcile(orders=client.orders(), state_store=store)
        click.echo(
            f"reconciled positions={len(reconciled.positions)} "
            f"open_orders={len(reconciled.open_orders)} drift={reconciled.drift_symbols}"
        )
        broker = kite_broker
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
        override_qty=order_qty,
        state_store=store,
    )
    try:
        result = asyncio.run(loop.run())
    finally:
        if store is not None:
            store.close()
    click.echo(
        f"live signals={result.stats.signals_seen} orders={result.stats.orders_placed} "
        f"qty_cap={order_qty} open={result.open_positions}"
    )


def _print_state_summary(store: OrderStateStore, path: Path) -> None:
    records = store.list_all()
    counts: dict[str, int] = {}
    for record in records:
        counts[record.state.value] = counts.get(record.state.value, 0) + 1
    if not counts:
        click.echo(f"state_db={path} records=0")
        return
    summary = " ".join(f"{state}={n}" for state, n in sorted(counts.items()))
    click.echo(f"state_db={path} records={len(records)} {summary}")


def _build_feed(
    *,
    bars: str | None,
    bars_count: int,
    live_feed: str,
    dry_run: bool,
    allow_replay_live: bool,
    settings: AppSettings,
) -> ReplayFeed | LiveKiteFeed:
    """Construct the bar source for ``live`` honouring safety guards.

    TRADEOFF: The kite live feed is constructed lazily here so dry-run
    paths don't need credentials. When credentials are missing we either
    fall back to replay (dry-run) or hard-fail (``--live-feed kite``).
    """
    if live_feed == "kite":
        if not settings.kite_configured():
            click.echo(
                "live --live-feed kite requires KITE_API_KEY and KITE_ACCESS_TOKEN"
            )
            raise SystemExit(1)
        client = KiteClient.from_settings(settings)
        return LiveKiteFeed(kite_client=client)

    if not dry_run:
        if bars is None:
            click.echo(
                "live --no-dry-run requires a real bar source; "
                "pass --live-feed kite (preferred) or use --dry-run"
            )
            raise SystemExit(1)
        if not allow_replay_live:
            click.echo(
                "live --no-dry-run with --bars requires --allow-replay-live; "
                "refusing to place real orders against replay data"
            )
            raise SystemExit(1)
        click.echo(
            "WARNING: live --no-dry-run reading from CSV/replay; "
            "real orders may be placed against stale data"
        )

    frame = (
        ReplayFeed(Path(bars)).to_dataframe()
        if bars is not None
        else make_synthetic_bars(bars_count, seed=42)
    )
    return ReplayFeed(frame)


@cli.command("flatten")
@click.option("--config", type=click.Path(), default=None)
@click.option(
    "--state-db",
    "state_db",
    type=click.Path(),
    default=None,
    help="Path to the persistent OrderStateStore DuckDB file "
    "(default: settings.state_db_path).",
)
def flatten_cmd(config: str | None, state_db: str | None) -> None:
    """Cancel tracked GTTs and square-off every open Kite position at market.

    Useful as a kill-switch recovery path: drop a ``KILL`` file to halt new
    entries, then run ``python -m orchestrator.main flatten`` to close
    everything down.
    """
    settings = load_settings(Path(config) if config else None)
    if not settings.kite_configured():
        click.echo("flatten requires KITE_API_KEY and KITE_ACCESS_TOKEN")
        raise SystemExit(1)
    state_db_path = Path(state_db) if state_db else settings.state_db_path
    store = OrderStateStore(state_db_path)
    try:
        _print_state_summary(store, state_db_path)
        client = KiteClient.from_settings(settings)
        broker = KiteBroker(client, settings=settings, state_store=store)
        try:
            broker.flatten_all()
        except FlattenIncomplete as exc:
            open_symbols = ",".join(p.symbol for p in exc.open_positions)
            click.echo(
                f"flatten incomplete after {exc.attempts} polls: "
                f"open={open_symbols}"
            )
            raise SystemExit(2) from exc
        click.echo("flatten complete")
    finally:
        store.close()


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


@cli.command("dashboard")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True)
@click.option("--config", type=click.Path(), default=None)
@click.option(
    "--i-know-what-im-doing",
    "force_non_localhost",
    is_flag=True,
    hidden=True,
    help="Allow binding to non-localhost host. DANGEROUS — no auth.",
)
def dashboard_cmd(
    host: str,
    port: int,
    config: str | None,
    force_non_localhost: bool,
) -> None:
    """Run the localhost dashboard at http://127.0.0.1:<port>.

    Single-user, no auth. Refuses non-localhost binds unless
    ``--i-know-what-im-doing`` is passed.
    """
    del config  # currently informational; dashboard loads config/config.yaml itself
    if host not in {"127.0.0.1", "localhost"} and not force_non_localhost:
        click.echo(f"refusing to bind {host}; localhost only by default")
        raise SystemExit(1)
    import uvicorn

    click.echo(f"dashboard running at http://{host}:{port}")
    uvicorn.run(
        "dashboard.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
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


@cli.command("screener")
@click.option(
    "--universe",
    "universe_path",
    type=click.Path(),
    default=None,
    help="Path to universe YAML (default: config/universe.yaml then config/universe.example.yaml).",
)
@click.option(
    "--provider",
    type=click.Choice(list(PROVIDER_OPTIONS)),
    default="stub",
    show_default=True,
    help="LLM provider. 'stub' returns DEFAULT_FORMULA — useful for offline testing.",
)
@click.option(
    "--fetch-missing/--no-fetch-missing",
    default=False,
    help="On-demand Kite fetch for symbols without local bars (requires Kite credentials).",
)
@click.option("--bars-back", default=200, show_default=True, help="Bars to load per symbol.")
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
)
@click.option("--config", type=click.Path(), default=None, help="Path to config YAML.")
@click.option(
    "--dashboard-db",
    "dashboard_db",
    type=click.Path(),
    default=None,
    help="DuckDB file for screener_runs/screener_picks (default: dashboard.duckdb).",
)
@click.option(
    "--notes",
    default="",
    help="Free-form notes added to MarketContext (e.g. trader observations).",
)
@click.option(
    "--index-summary",
    default="No external index summary provided.",
    help="Short market-context blurb passed to the LLM (regime hint).",
)
def screener_cmd(
    universe_path: str | None,
    provider: str,
    fetch_missing: bool,
    bars_back: int,
    output: str,
    config: str | None,
    dashboard_db: str | None,
    notes: str,
    index_summary: str,
) -> None:
    """Run the LLM screener once and persist + print the result.

    Examples:

        \b
        # Offline smoke test (no API keys, no network):
        python -m orchestrator.main screener --provider stub
    """
    settings = load_settings(Path(config) if config else None)
    try:
        universe = load_universe(Path(universe_path) if universe_path else None)
    except FileNotFoundError as exc:
        click.echo(f"universe not found: {exc}", err=True)
        raise SystemExit(1) from exc
    except ValueError as exc:
        click.echo(f"universe invalid: {exc}", err=True)
        raise SystemExit(1) from exc

    db_path = (
        Path(dashboard_db)
        if dashboard_db
        else settings.state_db_path.parent / "dashboard.duckdb"
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(SCREENER_RUNS_SCHEMA)
    conn.execute(SCREENER_PICKS_SCHEMA)
    try:
        store = ScreenerStore(conn)
        service = ScreenerService(store, settings)
        provider_name: ScreenerProviderName = provider  # type: ignore[assignment]
        ctx = MarketContext(
            as_of=datetime.now(tz=IST),
            recent_index_summary=index_summary,
            notes=notes,
        )
        try:
            record = asyncio.run(
                service.run(
                    provider_name=provider_name,
                    market_context=ctx,
                    fetch_missing=fetch_missing,
                    bars_back=bars_back,
                    universe=universe,
                )
            )
        except ValueError as exc:
            click.echo(f"screener config error: {exc}", err=True)
            raise SystemExit(1) from exc
    finally:
        conn.close()

    if output == "json":
        click.echo(render_run_record_json(record))
    else:
        click.echo(render_run_record_table(record))


@cli.command("tuner")
@click.option(
    "--provider",
    type=click.Choice(list(TUNER_PROVIDER_OPTIONS)),
    default="stub",
    show_default=True,
)
@click.option("--lookback-days", default=30, show_default=True)
@click.option("--notes", default="", help="Operator notes passed to the LLM.")
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
)
@click.option("--config", type=click.Path(), default=None)
@click.option(
    "--dashboard-db",
    "dashboard_db",
    type=click.Path(),
    default=None,
)
def tuner_cmd(
    provider: str,
    lookback_days: int,
    notes: str,
    output: str,
    config: str | None,
    dashboard_db: str | None,
) -> None:
    """Review recent backtest trades and propose strategy adjustments.

    Examples:

        \b
        python -m orchestrator.main tuner --provider stub
    """
    settings = load_settings(Path(config) if config else None)
    db_path = (
        Path(dashboard_db)
        if dashboard_db
        else settings.state_db_path.parent / "dashboard.duckdb"
    )
    state = AppState.build(
        settings=settings,
        dashboard_db_path=db_path,
    )
    try:
        from dashboard.services.tuner_service import TunerService

        service = TunerService(state)
        provider_name: TunerProviderName = provider  # type: ignore[assignment]
        record = asyncio.run(
            service.run(
                provider_name=provider_name,
                notes=notes,
                lookback_days=lookback_days,
            )
        )
    finally:
        state.close()

    if output == "json":
        click.echo(render_run_json(record))
    else:
        click.echo(render_run_table(record))


if __name__ == "__main__":
    cli()
