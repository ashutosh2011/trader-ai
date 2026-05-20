# Strategies

Define trading logic by subclassing `Strategy` and implementing `on_bar(ctx) -> list[Signal]`.
Use `compute_indicators(ctx, required_indicators)` or `Strategy.precompute_indicators(ctx)` for
warmup-safe indicator values. Register strategies with `@register_strategy` and look them up via
`get_strategy("ema_crossover")`. See `examples/ema_crossover.py` for a full reference implementation.
