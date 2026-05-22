"""Universe loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from screener.universe import (
    FALLBACK_UNIVERSE_PATH,
    Universe,
    UniverseSymbol,
    load_universe,
)


def test_load_example_universe() -> None:
    universe = load_universe(FALLBACK_UNIVERSE_PATH)
    assert isinstance(universe, Universe)
    assert len(universe.symbols) >= 5
    assert isinstance(universe.symbols[0], UniverseSymbol)


def test_load_explicit_path(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text(
        "symbols:\n"
        "  - {symbol: \"FOO\", instrument_token: 1, exchange: \"NSE\"}\n"
        "  - {symbol: \"BAR\"}\n",
        encoding="utf-8",
    )
    universe = load_universe(p)
    assert [s.symbol for s in universe.symbols] == ["FOO", "BAR"]
    assert universe.symbols[0].instrument_token == 1
    assert universe.symbols[1].instrument_token is None
    assert universe.symbols[1].exchange == "NSE"


def test_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_universe(tmp_path / "does-not-exist.yaml")


def test_schema_violation_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    # missing symbols key
    p.write_text("name: nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid universe schema"):
        load_universe(p)


def test_top_level_not_mapping_raises(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping at the top level"):
        load_universe(p)


def test_yaml_syntax_error_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text("symbols: [\n  unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed to parse"):
        load_universe(p)


def test_empty_symbols_list_rejected(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text("symbols: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_universe(p)


def test_empty_symbol_name_rejected(tmp_path: Path) -> None:
    p = tmp_path / "u.yaml"
    p.write_text("symbols:\n  - {symbol: \"   \"}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_universe(p)
