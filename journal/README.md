# Journal

Structured JSONL event log and LLM daily review notebook.

```python
from journal.log import TradingJournal
from journal.notebook import build_notebook

journal = TradingJournal("logs/session.jsonl")
journal.write_signal(signal)
notebook = build_notebook(Path("logs/session.jsonl"))
```

Run `await notebook.write_summary(date.today(), Path("reviews/today.md"))` for a markdown summary.
