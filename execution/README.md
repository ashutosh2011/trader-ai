# Execution

`Broker` ABC with paper simulation and live Kite Connect. Restart reconciliation syncs broker state.

```python
from execution.paper import PaperBroker
from execution.kite import KiteBroker
from execution.reconciler import StateReconciler

broker = PaperBroker(settings=load_settings())
order = broker.place_bracket_order(signal, qty=10)
StateReconciler(broker).reconcile()
```

Client order IDs are deterministic from `(strategy_id, signal_ts, symbol)`. Live orders require Kite credentials and an inactive kill switch.
