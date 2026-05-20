from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from backtest.engine import BacktestResult, BacktestSummary, EquityPoint
from backtest.report import write_markdown_report

IST = ZoneInfo("Asia/Kolkata")


def test_write_markdown_empty_trades(tmp_path: Path) -> None:
    ts = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    result = BacktestResult(
        closed_trades=[],
        equity_curve=[EquityPoint(bar_index=0, timestamp=ts, equity=0.0)],
        summary=BacktestSummary(
            trade_count=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0.0,
        ),
    )
    path = write_markdown_report(tmp_path / "empty.md", result)
    text = path.read_text(encoding="utf-8")
    assert "_No trades._" in text
