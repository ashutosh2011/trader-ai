from analyst.providers.anthropic import parse_verdict_json


def test_parse_verdict_json_codeblock() -> None:
    raw = (
        '```json\n{"action": "SHRINK", "size_multiplier": 0.4, '
        '"confidence": 0.6, "rationale": "x"}\n```'
    )
    verdict = parse_verdict_json(raw, provider="anthropic", latency_ms=10)
    assert verdict.action == "SHRINK"
    assert verdict.size_multiplier == 0.4
    assert verdict.provider == "anthropic"
