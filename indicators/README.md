# Indicators

Vectorized technical indicators with TradingView-compatible formulas. Register via `@register_indicator` and look up by name. Custom examples live under `indicators/custom/` (see `indicators/custom/README.md`).

```python
import pandas as pd
import indicators  # registers builtins

from indicators.registry import get_indicator

bars = pd.DataFrame({"open": [...], "high": [...], "low": [...], "close": [...], "volume": [...]})
rsi_cls = get_indicator("rsi")
rsi = rsi_cls(period=14).compute(bars)
print(rsi.iloc[-1], rsi_cls(period=14).warmup())
```
