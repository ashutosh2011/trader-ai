"""Unit tests for the dashboard's ``.env`` writer helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.services.env_writer import delete_env_var, upsert_env_var


def test_upsert_creates_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    upsert_env_var(env, "ANTHROPIC_API_KEY", "sk-anth-secret")
    assert env.read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=sk-anth-secret\n"


def test_upsert_replaces_existing_line(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OTHER=value\nANTHROPIC_API_KEY=old\nFOO=bar\n", encoding="utf-8")
    upsert_env_var(env, "ANTHROPIC_API_KEY", "new-secret")
    text = env.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=new-secret" in text
    assert "OTHER=value" in text
    assert "FOO=bar" in text
    assert text.count("ANTHROPIC_API_KEY=") == 1


def test_upsert_appends_when_missing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OTHER=value\n", encoding="utf-8")
    upsert_env_var(env, "OPENAI_API_KEY", "sk-openai")
    text = env.read_text(encoding="utf-8")
    assert text.endswith("OPENAI_API_KEY=sk-openai\n")
    assert "OTHER=value" in text


def test_upsert_quotes_values_with_spaces(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    upsert_env_var(env, "FOO", "has space and $stuff")
    text = env.read_text(encoding="utf-8")
    assert 'FOO="has space and $stuff"' in text


def test_upsert_rejects_bad_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    with pytest.raises(ValueError):
        upsert_env_var(env, "bad-key", "value")


def test_delete_removes_line(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\nGOOGLE_API_KEY=secret\nB=2\n", encoding="utf-8")
    removed = delete_env_var(env, "GOOGLE_API_KEY")
    assert removed is True
    assert env.read_text(encoding="utf-8") == "A=1\nB=2\n"


def test_delete_missing_returns_false(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\n", encoding="utf-8")
    assert delete_env_var(env, "MISSING") is False


def test_delete_no_file_returns_false(tmp_path: Path) -> None:
    env = tmp_path / "nonexistent.env"
    assert delete_env_var(env, "FOO") is False
