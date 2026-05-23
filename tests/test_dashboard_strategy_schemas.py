"""Schema-drift + JSON round-trip tests for the backtest form schemas."""

from __future__ import annotations

import inspect

import pytest

from dashboard.services.strategy_schemas import (
    STRATEGY_SCHEMAS,
    ParamSpec,
    StrategySchema,
    all_schemas,
    get_schema,
    strategy_param_keys,
    to_json_dict,
)
from strategies.registry import get_strategy, list_strategies


def test_every_registered_strategy_has_schema() -> None:
    # If somebody registers a sixth strategy without adding a schema, the
    # backtest form would silently lose the ability to configure it.
    registered = set(list_strategies())
    schema_ids = set(STRATEGY_SCHEMAS)
    assert registered == schema_ids, (
        f"schema drift: missing={registered - schema_ids}, "
        f"extra={schema_ids - registered}"
    )


@pytest.mark.parametrize("strategy_id", list(STRATEGY_SCHEMAS))
def test_schema_param_names_match_constructor(strategy_id: str) -> None:
    schema = STRATEGY_SCHEMAS[strategy_id]
    sig = inspect.signature(get_strategy(strategy_id).__init__)
    allowed = {name for name in sig.parameters if name not in {"self", "symbol"}}
    declared = {p.name for p in schema.params}
    assert declared <= allowed, (
        f"{strategy_id}: schema declares params not on ctor: {declared - allowed}"
    )
    # Each declared default also has to match the constructor's default so
    # the UI placeholder reflects reality.
    for param in schema.params:
        ctor_default = sig.parameters[param.name].default
        assert ctor_default == param.default, (
            f"{strategy_id}.{param.name}: schema default {param.default} "
            f"differs from ctor default {ctor_default!r}"
        )


@pytest.mark.parametrize("strategy_id", list(STRATEGY_SCHEMAS))
def test_param_bounds_are_sane(strategy_id: str) -> None:
    schema = STRATEGY_SCHEMAS[strategy_id]
    for param in schema.params:
        assert param.min < param.max, f"{strategy_id}.{param.name} bad bounds"
        assert param.step > 0, f"{strategy_id}.{param.name} step must be > 0"
        assert param.label, f"{strategy_id}.{param.name} missing label"
        assert param.help, f"{strategy_id}.{param.name} missing help"
        assert param.type in ("int", "float")
        assert param.min <= param.default <= param.max


def test_to_json_dict_roundtrip() -> None:
    payload = to_json_dict()
    for sid, schema in STRATEGY_SCHEMAS.items():
        entry = payload[sid]
        assert entry["id"] == sid
        assert entry["label"] == schema.label
        assert entry["summary"] == schema.summary
        names = [p["name"] for p in entry["params"]]
        assert names == [p.name for p in schema.params]
        # Every value in the JSON is itself JSON-serialisable.
        for spec in entry["params"]:
            assert set(spec.keys()) >= {
                "name", "label", "type", "default", "min", "max", "step", "help",
            }


def test_param_spec_to_json_is_plain_dict() -> None:
    spec = ParamSpec(
        name="x", label="X", type="int", default=1, min=0, max=10, step=1, help="h",
    )
    out = spec.to_json()
    assert out == {
        "name": "x", "label": "X", "type": "int", "default": 1,
        "min": 0, "max": 10, "step": 1, "help": "h",
    }


def test_get_schema_unknown_returns_none() -> None:
    assert get_schema("does_not_exist") is None


def test_all_schemas_sorted_by_label() -> None:
    schemas = all_schemas()
    labels = [s.label.lower() for s in schemas]
    assert labels == sorted(labels)


def test_strategy_param_keys_matches_ctor() -> None:
    for sid in list_strategies():
        keys = strategy_param_keys(sid)
        sig = inspect.signature(get_strategy(sid).__init__)
        assert keys == {n for n in sig.parameters if n not in {"self", "symbol"}}


def test_strategy_schema_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    schema = next(iter(STRATEGY_SCHEMAS.values()))
    assert isinstance(schema, StrategySchema)
    with pytest.raises(FrozenInstanceError):
        schema.label = "x"  # type: ignore[misc]
