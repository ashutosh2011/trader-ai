# Custom indicators

Copy `example_momentum.py` to add your own logic. Register with `@register_indicator` and import the module.

```python
import indicators.custom  # registers price_momentum
from indicators.registry import get_indicator

momentum = get_indicator("price_momentum")(period=10).compute(bars)
```

Replace the formula in your copy — the example uses `close - close.shift(n)` only as a pattern.
