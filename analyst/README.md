# Analyst

Advisory LLM layer: `Analyst.analyze(signal, ctx)` returns a `Verdict` (APPROVE, VETO, SHRINK).
On timeout or error, falls back to APPROVE at 0.7× with provider `fallback`.

## Usage

```python
from analyst.analyst import Analyst
from analyst.providers.mock import MockLLMProvider

provider = MockLLMProvider('{"action": "APPROVE", "size_multiplier": 0.8, ...}')
verdict = await Analyst(provider).analyze(signal, ctx)
```

Providers: `MockLLMProvider`, `AnthropicProvider`, `OpenAIProvider`, `GoogleProvider` (API keys via env).

## A/B test

```bash
python -m orchestrator.main ab-test --bars-count 500
python -m orchestrator.main ab-test --veto
```
