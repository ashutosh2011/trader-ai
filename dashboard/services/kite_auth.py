"""Kite Connect auth helper used by the ``/kite`` page.

Wraps :mod:`orchestrator.kite_login` with dashboard-friendly status helpers
(token presence + ``.env`` mtime / day-stale check). The token write path
delegates to :func:`orchestrator.kite_login.update_env_access_token`,
which already does an in-place line replacement preserving every other
``.env`` line.

TRADEOFF: We do not detect IST trading-day boundaries precisely (e.g.
the actual EOD expiry time Zerodha enforces). We compare the file's
local date against today's local date — a token written yesterday
flags as stale even at 00:01 today, which is the right user behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import structlog
from kiteconnect.exceptions import KiteException

from orchestrator.kite_login import (
    exchange_request_token,
    kite_login_url,
    update_env_access_token,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class KiteTokenStatus:
    """Status of the current Kite access token on disk."""

    has_token: bool
    env_mtime: datetime | None
    is_day_stale: bool


class KiteAuthService:
    """Build login URLs and exchange request tokens for the dashboard."""

    def __init__(
        self,
        *,
        env_path: Path,
        api_key: str | None,
        api_secret: str | None,
        access_token: str | None,
    ) -> None:
        """Construct the service bound to a specific ``.env`` path + creds."""
        self._env_path = env_path
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = access_token

    @property
    def api_key(self) -> str | None:
        """Current Kite API key (read at construction time)."""
        return self._api_key

    @property
    def env_path(self) -> Path:
        """Path to the ``.env`` we will write the token to."""
        return self._env_path

    def is_configured(self) -> bool:
        """Return whether API key + secret are both present."""
        return bool(self._api_key) and bool(self._api_secret)

    def login_url(self) -> str:
        """Build the Kite login URL the user opens in a browser.

        Raises:
            ValueError: If no API key is configured.
        """
        if not self._api_key:
            msg = "KITE_API_KEY missing; set it in .env first"
            raise ValueError(msg)
        return kite_login_url(self._api_key)

    def token_status(self) -> KiteTokenStatus:
        """Inspect the current token + ``.env`` mtime for the UI banner."""
        has_token = bool(self._access_token)
        mtime: datetime | None = None
        if self._env_path.is_file():
            mtime = datetime.fromtimestamp(self._env_path.stat().st_mtime).astimezone()
        stale = False
        if has_token and mtime is not None:
            stale = mtime.date() < date.today()
        return KiteTokenStatus(has_token=has_token, env_mtime=mtime, is_day_stale=stale)

    def exchange(self, request_token: str) -> str:
        """Exchange ``request_token`` for an access token and persist it.

        Args:
            request_token: One-time token from the Kite redirect URL.

        Returns:
            The newly-issued access token (also written to ``.env``).

        Raises:
            ValueError: If credentials are missing or ``request_token`` is empty.
            FileNotFoundError: If the ``.env`` file does not exist.
        """
        if not self.is_configured():
            msg = "KITE_API_KEY / KITE_API_SECRET missing"
            raise ValueError(msg)
        token = request_token.strip()
        if not token:
            msg = "request_token must not be empty"
            raise ValueError(msg)
        assert self._api_key is not None and self._api_secret is not None
        try:
            access_token = exchange_request_token(self._api_key, self._api_secret, token)
        except KiteException as exc:
            logger.warning(
                "dashboard_kite_token_exchange_failed",
                error=str(exc),
                env=str(self._env_path),
            )
            msg = (
                "Kite rejected the request token. Check that the request_token was "
                "copied from the login redirect for this exact API key, has not "
                "expired or already been used, and that KITE_API_SECRET in .env "
                "matches the Kite developer console."
            )
            raise ValueError(msg) from exc
        update_env_access_token(access_token, self._env_path)
        logger.warning(
            "dashboard_kite_token_written",
            env=str(self._env_path),
        )
        self._access_token = access_token
        return access_token
