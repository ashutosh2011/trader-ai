"""Schema validation tests for the screener formula models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from screener.schema import (
    CompareTo,
    IndicatorFilter,
    PriceChangeFilter,
    ScreenerFormula,
    VolumeFilter,
)


def _valid_formula_payload() -> dict[str, object]:
    return {
        "name": "Test",
        "timeframe": "day",
        "side_bias": "long",
        "rationale": "test",
        "filters": [
            {
                "type": "indicator",
                "indicator": "rsi",
                "params": {"period": 14},
                "op": "<",
                "value": 30.0,
            }
        ],
    }


def test_valid_formula_parses() -> None:
    payload = _valid_formula_payload()
    formula = ScreenerFormula.model_validate(payload)
    assert formula.name == "Test"
    assert formula.timeframe == "day"
    assert formula.side_bias == "long"
    assert len(formula.filters) == 1


def test_formula_with_compare_to_parses() -> None:
    payload = _valid_formula_payload()
    payload["filters"] = [
        {
            "type": "indicator",
            "indicator": "close",
            "op": ">",
            "compare_to": {"indicator": "sma", "params": {"period": 50}},
        }
    ]
    formula = ScreenerFormula.model_validate(payload)
    assert isinstance(formula.filters[0], IndicatorFilter)
    assert formula.filters[0].compare_to is not None
    assert formula.filters[0].compare_to.indicator == "sma"


def test_indicator_filter_rejects_both_value_and_compare_to() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        IndicatorFilter(
            indicator="rsi",
            op=">",
            value=30.0,
            compare_to=CompareTo(indicator="sma", params={"period": 50}),
        )


def test_indicator_filter_rejects_neither_value_nor_compare_to() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        IndicatorFilter(indicator="rsi", op=">")


def test_indicator_filter_rejects_empty_indicator() -> None:
    with pytest.raises(ValidationError, match="indicator must be non-empty"):
        IndicatorFilter(indicator="   ", op=">", value=1.0)


def test_compare_to_rejects_empty_indicator() -> None:
    with pytest.raises(ValidationError, match="compare_to.indicator must be non-empty"):
        CompareTo(indicator="", params={})


def test_compare_to_cannot_nest_another_compare_to() -> None:
    payload = _valid_formula_payload()
    payload["filters"] = [
        {
            "type": "indicator",
            "indicator": "close",
            "op": ">",
            "compare_to": {
                "indicator": "sma",
                "params": {"period": 50},
                # extra field — schema is extra="forbid"
                "compare_to": {"indicator": "ema"},
            },
        }
    ]
    with pytest.raises(ValidationError):
        ScreenerFormula.model_validate(payload)


def test_empty_filters_rejected() -> None:
    payload = _valid_formula_payload()
    payload["filters"] = []
    with pytest.raises(ValidationError):
        ScreenerFormula.model_validate(payload)


def test_bad_timeframe_rejected() -> None:
    payload = _valid_formula_payload()
    payload["timeframe"] = "1h"
    with pytest.raises(ValidationError):
        ScreenerFormula.model_validate(payload)


def test_bad_side_bias_rejected() -> None:
    payload = _valid_formula_payload()
    payload["side_bias"] = "flat"
    with pytest.raises(ValidationError):
        ScreenerFormula.model_validate(payload)


def test_bad_op_rejected() -> None:
    payload = _valid_formula_payload()
    payload["filters"][0]["op"] = "!="  # type: ignore[index]
    with pytest.raises(ValidationError):
        ScreenerFormula.model_validate(payload)


def test_extra_fields_rejected() -> None:
    payload = _valid_formula_payload()
    payload["unknown_top_level"] = "nope"
    with pytest.raises(ValidationError):
        ScreenerFormula.model_validate(payload)


def test_volume_filter_requires_exactly_one_form() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        VolumeFilter(op=">", value=100.0, value_x_avg=1.5)
    with pytest.raises(ValidationError, match="exactly one"):
        VolumeFilter(op=">")


def test_volume_filter_avg_window_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        VolumeFilter(op=">", value_x_avg=1.5, avg_window=0)


def test_price_change_filter_requires_positive_window() -> None:
    with pytest.raises(ValidationError):
        PriceChangeFilter(window=0, op=">", value_pct=1.0)


def test_name_must_be_non_empty() -> None:
    payload = _valid_formula_payload()
    payload["name"] = "   "
    with pytest.raises(ValidationError, match="name must be non-empty"):
        ScreenerFormula.model_validate(payload)


def test_volume_filter_payload_round_trips() -> None:
    payload = _valid_formula_payload()
    payload["filters"] = [
        {"type": "volume", "op": ">", "value_x_avg": 1.5, "avg_window": 20}
    ]
    formula = ScreenerFormula.model_validate(payload)
    dumped = formula.model_dump()
    assert dumped["filters"][0]["type"] == "volume"
    again = ScreenerFormula.model_validate(dumped)
    assert again == formula


def test_price_change_filter_payload_round_trips() -> None:
    payload = _valid_formula_payload()
    payload["filters"] = [
        {"type": "price_change", "window": 5, "op": ">", "value_pct": 3.0}
    ]
    formula = ScreenerFormula.model_validate(payload)
    dumped = formula.model_dump_json()
    again = ScreenerFormula.model_validate_json(dumped)
    assert again.filters[0] == formula.filters[0]
