"""Daily trading journal review via LLM."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import structlog

from analyst.provider import LLMProvider
from analyst.providers.mock import MockLLMProvider

logger = structlog.get_logger(__name__)

DAILY_REVIEW_PROMPT = """You are a trading coach. Summarize the following JSONL journal
entries for {trade_date} in markdown. Cover: signals emitted, risk rejections,
analyst vetoes/approvals, orders placed, and lessons. Be concise (under 400 words).

Journal:
{entries}
"""


class DailyNotebook:
    """Generate a markdown daily summary from JSONL journal + LLM."""

    def __init__(self, journal_path: Path, provider: LLMProvider) -> None:
        self._path = journal_path
        self._provider = provider

    async def summarize_day(self, trade_date: date) -> str:
        """Read journal events for ``trade_date`` and return markdown summary."""
        entries = _load_entries_for_date(self._path, trade_date)
        if not entries:
            return f"# Daily review {trade_date}\n\nNo journal entries for this date.\n"
        prompt = DAILY_REVIEW_PROMPT.format(
            trade_date=trade_date.isoformat(),
            entries=json.dumps(entries, indent=2, default=str),
        )
        raw = await self._provider.complete(prompt)
        logger.info("daily_notebook_complete", date=str(trade_date), chars=len(raw))
        return raw.strip() + "\n"

    async def write_summary(self, trade_date: date, output_path: Path) -> Path:
        """Write markdown summary to ``output_path``."""
        content = await self.summarize_day(trade_date)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path


def build_notebook(
    journal_path: Path,
    *,
    provider: LLMProvider | None = None,
) -> DailyNotebook:
    """Factory: uses mock provider when none supplied."""
    llm = provider or MockLLMProvider(
        "# Daily Review\n\nMock summary: no live LLM configured.\n",
        name="mock",
    )
    return DailyNotebook(journal_path, llm)


def _load_entries_for_date(path: Path, trade_date: date) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    prefix = trade_date.isoformat()
    entries: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ts = str(record.get("ts", ""))
        if prefix in ts:
            entries.append(record)
    return entries
