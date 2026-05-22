"""Post-trade strategy tuner: LLM reviews results and proposes changes.

The tuner is advisory like the screener and analyst — it emits structured
JSON recommendations that a human applies from the dashboard (or CLI
``tuner apply`` in a later step). It does not auto-modify live trading.
"""
