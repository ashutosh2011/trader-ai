"""Shared LLM JSON extraction helpers.

Lifted from :mod:`analyst.analyst` so the screener can reuse the same
layered extraction strategy without duplicating regex logic. The analyst
re-exports these names for backward compatibility.
"""

from __future__ import annotations


def strip_code_fences(text: str) -> str:
    """Strip triple-backtick fences (with optional language tag) if present."""
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def _find_first_balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring at any nesting depth.

    Uses brace counting so the screener's 3+ level nested formulas are
    extracted whole, instead of returning an innermost-only match.
    Quoted braces inside JSON strings are honoured (with backslash
    escape handling) so ``"{"`` in a string value won't trip the
    counter.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def extract_json_object(raw: str) -> str:
    """Extract a JSON object substring from raw LLM text.

    Layered strategy:
        1. Strip surrounding whitespace.
        2. If the text is wrapped in a triple-backtick fence, unwrap it.
        3. If the (possibly unwrapped) text starts with ``{`` and ends
           with ``}``, return it verbatim.
        4. Otherwise, return the first balanced ``{...}`` substring,
           using brace counting that supports arbitrary nesting depth.
        5. Raise :class:`ValueError` if no candidate JSON object is found.

    Args:
        raw: Raw text returned by an LLM provider.

    Returns:
        A JSON object substring suitable for :func:`json.loads`.

    Raises:
        ValueError: If no balanced JSON object can be located.
    """
    text = raw.strip()
    if not text:
        msg = "empty LLM response"
        raise ValueError(msg)
    if text.startswith("```"):
        fenced = strip_code_fences(text)
        if fenced and fenced != text:
            text = fenced
    if text.startswith("{") and text.endswith("}"):
        return text
    found = _find_first_balanced_object(text)
    if found is not None:
        return found
    msg = "no JSON object found in LLM response"
    raise ValueError(msg)


__all__ = ["extract_json_object", "strip_code_fences"]
