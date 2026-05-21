"""Localhost-only dashboard for the tradebot project.

Exposes a FastAPI app at :data:`dashboard.server.app` that ships with
HTMX/Tailwind/Chart.js (via CDN) and renders Jinja2 templates server-side.

TRADEOFF: This dashboard is single-user and assumes localhost. There is
**no authentication** — anyone who can connect to the bound port can flip
the kill switch, flatten positions, or rewrite config. Bind to
``127.0.0.1`` (the default) and do not expose to the network.
"""

from dashboard.state import AppState, get_app_state

__all__ = ["AppState", "get_app_state"]
