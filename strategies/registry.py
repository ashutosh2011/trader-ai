"""Strategy registration and lookup by strategy id."""

from strategies.base import Strategy

_STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(cls: type[Strategy]) -> type[Strategy]:
    """Register a strategy class by its ``id`` class variable."""
    _STRATEGY_REGISTRY[cls.id] = cls
    return cls


def get_strategy(strategy_id: str) -> type[Strategy]:
    """Look up a registered strategy class by id."""
    if strategy_id not in _STRATEGY_REGISTRY:
        msg = f"strategy not registered: {strategy_id}"
        raise KeyError(msg)
    return _STRATEGY_REGISTRY[strategy_id]


def list_strategies() -> list[str]:
    """Return registered strategy ids in sorted order."""
    return sorted(_STRATEGY_REGISTRY)


import strategies.examples as _examples  # noqa: F401 — register examples on import
