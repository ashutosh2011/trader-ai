from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from journal.log import TradingJournal
from risk.manager import RiskDecision
from tests.test_risk_manager import _signal

IST = ZoneInfo("Asia/Kolkata")


def test_journal_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = TradingJournal(path)
    ts = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    journal.write_signal(_signal(ts))
    journal.write_risk_decision(RiskDecision(approved=True, reason="ok"), "SYNTH")
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert "signal" in lines[0]
