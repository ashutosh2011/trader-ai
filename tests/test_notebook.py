import json
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from analyst.providers.mock import MockLLMProvider
from journal.notebook import DailyNotebook

IST = ZoneInfo("Asia/Kolkata")


@pytest.mark.asyncio
async def test_daily_notebook_mock(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    record = {
        "event": "signal",
        "ts": "2024-01-02T10:00:00+05:30",
        "signal": {"symbol": "SYNTH"},
    }
    journal.write_text(json.dumps(record) + "\n", encoding="utf-8")
    provider = MockLLMProvider("# Review\n\nGood day.\n")
    notebook = DailyNotebook(journal, provider)
    md = await notebook.summarize_day(date(2024, 1, 2))
    assert "Review" in md


@pytest.mark.asyncio
async def test_daily_notebook_empty(tmp_path: Path) -> None:
    journal = tmp_path / "empty.jsonl"
    journal.write_text("", encoding="utf-8")
    notebook = DailyNotebook(journal, MockLLMProvider("x"))
    md = await notebook.summarize_day(date(2024, 1, 1))
    assert "No journal entries" in md
