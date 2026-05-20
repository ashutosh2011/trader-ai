from indicators.base import Indicator

_INDICATOR_REGISTRY: dict[str, type[Indicator]] = {}


def register_indicator(cls: type[Indicator]) -> type[Indicator]:
    """Register an indicator class by its ``name`` class variable."""
    _INDICATOR_REGISTRY[cls.name] = cls
    return cls


def get_indicator(name: str) -> type[Indicator]:
    """Look up a registered indicator class by name."""
    if name not in _INDICATOR_REGISTRY:
        msg = f"indicator not registered: {name}"
        raise KeyError(msg)
    return _INDICATOR_REGISTRY[name]


def list_indicators() -> list[str]:
    """Return registered indicator names in sorted order."""
    return sorted(_INDICATOR_REGISTRY)


import indicators.builtin as _builtin  # noqa: F401 — register builtins on import
import indicators.custom as _custom  # noqa: F401 — register custom examples on import
