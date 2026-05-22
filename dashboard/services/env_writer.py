"""Safe in-place ``.env`` writer used by the LLM settings page.

The dashboard's "/llm" page lets the operator paste API keys; those keys
must land in ``.env`` (which is gitignored) without disturbing the rest
of the file. This module mirrors the line-replacement approach used by
:mod:`orchestrator.kite_login.update_env_access_token` but generalises
it to arbitrary keys + delete support.

TRADEOFF: We treat ``.env`` as line-oriented text — no YAML, no quoting
rules beyond a conservative double-quote wrap when the value contains a
space or shell special character. This matches how the existing
``KITE_*`` lines look and keeps the dashboard's surface tiny.
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_KEY_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_NEEDS_QUOTE = re.compile(r"[\s\"'$`]")


def _validate_key(key: str) -> None:
    if not _KEY_PATTERN.match(key):
        msg = f"invalid env key: {key!r}"
        raise ValueError(msg)


def _format_line(key: str, value: str) -> str:
    if value == "" or not _NEEDS_QUOTE.search(value):
        return f"{key}={value}"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{key}="{escaped}"'


def upsert_env_var(env_path: Path, key: str, value: str) -> None:
    """Set or replace ``key=value`` in the ``.env`` file at ``env_path``.

    Creates the file (and any missing parent directories) when it does
    not yet exist. Existing lines for other keys are preserved verbatim,
    including comments and blank lines.

    Args:
        env_path: Path to the ``.env`` file.
        key: Variable name. Must match ``[A-Z_][A-Z0-9_]*``.
        value: New value. The empty string writes ``KEY=`` (this is how
            the example .env represents an unset key).

    Raises:
        ValueError: If ``key`` is not a valid env variable name.
    """
    _validate_key(key)
    line = _format_line(key, value)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if not env_path.is_file():
        env_path.write_text(line + "\n", encoding="utf-8")
        logger.info("dashboard_env_created", path=str(env_path), key=key)
        return
    text = env_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(key)}=.*$", flags=re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(line, text, count=1)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += line + "\n"
    env_path.write_text(text, encoding="utf-8")
    logger.info("dashboard_env_upserted", path=str(env_path), key=key)


def delete_env_var(env_path: Path, key: str) -> bool:
    """Remove a ``KEY=...`` line entirely if present.

    Args:
        env_path: Path to the ``.env`` file.
        key: Variable name to drop.

    Returns:
        ``True`` if a matching line was removed, ``False`` when the file
        is missing or the key was absent.
    """
    _validate_key(key)
    if not env_path.is_file():
        return False
    text = env_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(key)}=.*\n?", flags=re.MULTILINE)
    new_text, count = pattern.subn("", text, count=1)
    if count == 0:
        return False
    env_path.write_text(new_text, encoding="utf-8")
    logger.info("dashboard_env_deleted", path=str(env_path), key=key)
    return True


__all__ = ["delete_env_var", "upsert_env_var"]
