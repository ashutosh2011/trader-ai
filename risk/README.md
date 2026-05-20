# Risk

`RiskManager` enforces pre-trade checks and position sizing from `config/settings.py`.

## Usage

```python
from config.settings import load_settings
from risk.manager import RiskManager, RiskState

settings = load_settings("config/config.yaml")
rm = RiskManager(settings)
state = RiskState(account_equity=100_000.0, open_positions=0)
decision = rm.pre_check(signal, ctx, state)
if decision.approved:
    qty = rm.size(signal, ctx, state, size_multiplier=1.0)
```

## Rules

- Kill switch: `./KILL` file or `KILL_SWITCH=1`
- Daily loss cap, max open positions
- No-trade windows (`HH:MM-HH:MM` in IST)
- Options expiry-day entry cutoff
- Sizing: `atr_based` (risk / stop distance) or `fixed_pct`
