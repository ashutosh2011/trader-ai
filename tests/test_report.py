from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from backtest.engine import BacktestResult, BacktestSummary, ClosedTrade, EquityPoint
from backtest.report import generate_html_report, generate_markdown_report, write_html_report

IST = ZoneInfo("Asia/Kolkata")


def _minimal_result() -> BacktestResult:
    trade = ClosedTrade(
        symbol="SYNTH",
        side="LONG",
        entry_price=100.0,
        exit_price=105.0,
        qty=1,
        entry_bar=1,
        exit_bar=5,
        pnl=5.0,
        exit_reason="target",
    )
    ts = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    return BacktestResult(
        closed_trades=[trade],
        equity_curve=[EquityPoint(bar_index=0, timestamp=ts, equity=0.0)],
        summary=BacktestSummary(
            trade_count=1,
            winning_trades=1,
            losing_trades=0,
            total_pnl=5.0,
        ),
    )


def test_markdown_report_contains_keys() -> None:
    md = generate_markdown_report(_minimal_result(), title="Test Report")
    assert "# Test Report" in md
    assert "Sharpe Ratio" in md
    assert "SYNTH" in md
    assert "target" in md


def test_html_report_contains_keys(tmp_path: Path) -> None:
    html = generate_html_report(_minimal_result(), title="HTML Test")
    assert "<html" in html
    assert "Sharpe Ratio" in html
    assert "SYNTH" in html
    out = write_html_report(tmp_path / "report.html", _minimal_result())
    assert out.exists()
    assert "Performance Summary" in out.read_text(encoding="utf-8")
