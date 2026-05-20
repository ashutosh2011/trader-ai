"""Markdown and HTML backtest reports."""

import html
import json
from datetime import datetime
from pathlib import Path

from backtest.engine import BacktestResult, ClosedTrade
from backtest.metrics import PerformanceMetrics, compute_performance_metrics


def generate_markdown_report(
    result: BacktestResult,
    *,
    metrics: PerformanceMetrics | None = None,
    initial_capital: float = 100_000.0,
    title: str = "Backtest Report",
) -> str:
    """Build a Markdown report for a backtest result."""
    perf = metrics or compute_performance_metrics(result, initial_capital=initial_capital)
    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "",
        "## Performance Summary",
        "",
        _metrics_table_md(perf),
        "",
        "## Equity Curve",
        "",
        "```json",
        json.dumps(_equity_json(result), indent=2),
        "```",
        "",
        "## Trades",
        "",
        _trades_table_md(result.closed_trades),
        "",
    ]
    return "\n".join(lines)


def generate_html_report(
    result: BacktestResult,
    *,
    metrics: PerformanceMetrics | None = None,
    initial_capital: float = 100_000.0,
    title: str = "Backtest Report",
) -> str:
    """Build a self-contained HTML report (no external CSS/JS deps)."""
    perf = metrics or compute_performance_metrics(result, initial_capital=initial_capital)
    equity_json = html.escape(json.dumps(_equity_json(result)))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
    h1, h2 {{ color: #0d47a1; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #e3f2fd; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .pnl-pos {{ color: #2e7d32; }}
    .pnl-neg {{ color: #c62828; }}
    pre {{ background: #f5f5f5; padding: 1rem; overflow-x: auto; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Generated: {html.escape(datetime.now().astimezone().isoformat())}</p>
  <h2>Performance Summary</h2>
  {_metrics_table_html(perf)}
  <h2>Equity Curve (JSON)</h2>
  <pre>{equity_json}</pre>
  <h2>Equity Sparkline (ASCII)</h2>
  <pre>{html.escape(_ascii_equity_sparkline(result, initial_capital=initial_capital))}</pre>
  <h2>Trades</h2>
  {_trades_table_html(result.closed_trades)}
</body>
</html>
"""


def write_markdown_report(
    path: Path | str,
    result: BacktestResult,
    *,
    metrics: PerformanceMetrics | None = None,
    initial_capital: float = 100_000.0,
    title: str = "Backtest Report",
) -> Path:
    """Write Markdown report to ``path`` and return the resolved path."""
    out = Path(path)
    out.write_text(
        generate_markdown_report(
            result,
            metrics=metrics,
            initial_capital=initial_capital,
            title=title,
        ),
        encoding="utf-8",
    )
    return out.resolve()


def write_html_report(
    path: Path | str,
    result: BacktestResult,
    *,
    metrics: PerformanceMetrics | None = None,
    initial_capital: float = 100_000.0,
    title: str = "Backtest Report",
) -> Path:
    """Write HTML report to ``path`` and return the resolved path."""
    out = Path(path)
    out.write_text(
        generate_html_report(
            result,
            metrics=metrics,
            initial_capital=initial_capital,
            title=title,
        ),
        encoding="utf-8",
    )
    return out.resolve()


def _metrics_rows(perf: PerformanceMetrics) -> list[tuple[str, str]]:
    return [
        ("Sharpe Ratio", f"{perf.sharpe_ratio:.4f}"),
        ("Sortino Ratio", f"{perf.sortino_ratio:.4f}"),
        ("Max Drawdown", f"{perf.max_drawdown:.2f}"),
        ("Max Drawdown %", f"{perf.max_drawdown_pct:.2f}%"),
        ("Win Rate", f"{perf.win_rate_pct:.2f}%"),
        ("Average R", f"{perf.average_r:.4f}"),
        ("Expectancy", f"{perf.expectancy:.4f}"),
        ("Profit Factor", f"{perf.profit_factor:.4f}"),
        ("Total Return %", f"{perf.total_return_pct:.4f}%"),
        ("Total Trades", str(perf.total_trades)),
        ("Avg Win", f"{perf.avg_win:.2f}"),
        ("Avg Loss", f"{perf.avg_loss:.2f}"),
        ("Total PnL", f"{perf.total_pnl:.2f}"),
    ]


def _metrics_table_md(perf: PerformanceMetrics) -> str:
    rows = _metrics_rows(perf)
    header = "| Metric | Value |\n|--------|-------|"
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return f"{header}\n{body}"


def _metrics_table_html(perf: PerformanceMetrics) -> str:
    rows = _metrics_rows(perf)
    trs = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows
    )
    return (
        f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{trs}</tbody></table>"
    )


def _trades_table_md(trades: list[ClosedTrade]) -> str:
    if not trades:
        return "_No trades._"
    header = (
        "| # | Symbol | Side | Entry | Exit | Qty | PnL | Reason |\n"
        "|---|--------|------|-------|------|-----|-----|--------|"
    )
    lines = []
    for i, t in enumerate(trades, start=1):
        lines.append(
            f"| {i} | {t.symbol} | {t.side} | {t.entry_price:.2f} | {t.exit_price:.2f} | "
            f"{t.qty} | {t.pnl:.2f} | {t.exit_reason} |"
        )
    return f"{header}\n" + "\n".join(lines)


def _trades_table_html(trades: list[ClosedTrade]) -> str:
    if not trades:
        return "<p><em>No trades.</em></p>"
    rows = []
    for i, t in enumerate(trades, start=1):
        pnl_class = "pnl-pos" if t.pnl >= 0 else "pnl-neg"
        rows.append(
            "<tr>"
            f"<td>{i}</td><td>{html.escape(t.symbol)}</td><td>{html.escape(t.side)}</td>"
            f"<td>{t.entry_price:.2f}</td><td>{t.exit_price:.2f}</td>"
            f"<td>{t.qty}</td><td class='{pnl_class}'>{t.pnl:.2f}</td>"
            f"<td>{html.escape(t.exit_reason)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>#</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th>"
        "<th>Qty</th><th>PnL</th><th>Reason</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _equity_json(result: BacktestResult) -> list[dict[str, object]]:
    return [
        {
            "bar_index": p.bar_index,
            "timestamp": p.timestamp.isoformat(),
            "equity": p.equity,
        }
        for p in result.equity_curve
    ]


def _ascii_equity_sparkline(
    result: BacktestResult,
    *,
    initial_capital: float,
    width: int = 60,
) -> str:
    """Simple ASCII sparkline of portfolio value."""
    if not result.equity_curve:
        return "(empty)"
    values = [initial_capital + p.equity for p in result.equity_curve]
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return "_" * min(width, len(values))
    chars = " ._-:=+*#%@"
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    span = vmax - vmin
    line = "".join(
        chars[min(len(chars) - 1, int((v - vmin) / span * (len(chars) - 1)))] for v in sampled
    )
    return f"{line}\nmin={vmin:.2f} max={vmax:.2f}"
