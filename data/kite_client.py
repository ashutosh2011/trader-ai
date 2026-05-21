"""Kite Connect wrapper with rate limiting and token helpers."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import structlog
from pydantic import BaseModel, ConfigDict, Field

from config.settings import AppSettings, KiteConfig

IST = ZoneInfo("Asia/Kolkata")
logger = structlog.get_logger(__name__)

DEFAULT_MIN_INTERVAL_SEC = 0.34


class KiteSession(BaseModel):
    """Active Kite session metadata."""

    model_config = ConfigDict(frozen=True)

    api_key: str
    access_token: str
    login_time: datetime = Field(default_factory=lambda: datetime.now(tz=IST))


class RateLimiter:
    """Simple minimum-interval rate limiter for Kite REST calls."""

    def __init__(self, min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC) -> None:
        self._min_interval = min_interval_sec
        self._last_call = 0.0

    def wait(self) -> None:
        """Block until the next call is allowed."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


class KiteConnectProtocol(Protocol):
    """Subset of kiteconnect.KiteConnect used by :class:`KiteClient`."""

    def set_access_token(self, access_token: str) -> None: ...

    def historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
        continuous: bool = False,
        oi: bool = False,
    ) -> list[dict[str, Any]]: ...

    def orders(self) -> list[dict[str, Any]]: ...

    def positions(self) -> dict[str, list[dict[str, Any]]]: ...

    def place_order(self, **kwargs: Any) -> str: ...

    def cancel_order(self, variety: str, order_id: str) -> str: ...

    def place_gtt(self, **kwargs: Any) -> dict[str, Any]: ...

    def get_gtts(self) -> list[dict[str, Any]]: ...

    def delete_gtt(self, trigger_id: int) -> dict[str, Any]: ...


class KiteClient:
    """Wraps KiteConnect with rate limiting and credential validation.

    TRADEOFF: Access tokens expire daily; personal deployments must refresh
    manually via Kite login and set ``KITE_ACCESS_TOKEN`` in ``.env``.
    """

    def __init__(
        self,
        config: KiteConfig,
        *,
        kite: KiteConnectProtocol | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        if not config.api_key or not config.access_token:
            msg = "kite api_key and access_token are required"
            raise ValueError(msg)
        self._config = config
        self._kite = kite or _build_kite(config)
        self._rate = rate_limiter or RateLimiter()
        self._session = KiteSession(
            api_key=config.api_key,
            access_token=config.access_token,
        )
        logger.info("kite_client_ready", api_key_prefix=config.api_key[:4])

    @classmethod
    def from_settings(cls, settings: AppSettings) -> KiteClient:
        """Build client from application settings."""
        return cls(settings.kite)

    @property
    def session(self) -> KiteSession:
        return self._session

    @property
    def access_token(self) -> str:
        """Return the current access token (used by :class:`LiveKiteFeed`)."""
        return self._session.access_token

    @property
    def api_key(self) -> str:
        """Return the configured API key (used by :class:`LiveKiteFeed`)."""
        return self._session.api_key

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke a Kite API method with rate limiting."""
        self._rate.wait()
        return fn(*args, **kwargs)

    def historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV candles from Kite."""
        return cast(
            list[dict[str, Any]],
            self.call(
                self._kite.historical_data,
                instrument_token,
                from_date,
                to_date,
                interval,
            ),
        )

    def orders(self) -> list[dict[str, Any]]:
        """Return today's order book."""
        return cast(list[dict[str, Any]], self.call(self._kite.orders))

    def positions(self) -> dict[str, list[dict[str, Any]]]:
        """Return net/day positions."""
        return cast(dict[str, list[dict[str, Any]]], self.call(self._kite.positions))

    def place_order(self, **kwargs: Any) -> str:
        """Place an order; returns broker order id."""
        return cast(str, self.call(self._kite.place_order, **kwargs))

    def place_gtt(self, **kwargs: Any) -> dict[str, Any]:
        """Place a GTT (single or OCO two-leg); returns ``{"trigger_id": int}``.

        Accepts the kiteconnect ``place_gtt`` keyword arguments —
        ``trigger_type`` (``"single"`` or ``"two-leg"``), ``tradingsymbol``,
        ``exchange``, ``trigger_values``, ``last_price``, and ``orders``.
        """
        return cast(dict[str, Any], self.call(self._kite.place_gtt, **kwargs))

    def get_gtts(self) -> list[dict[str, Any]]:
        """Return all GTTs visible to the account (active, triggered, expired)."""
        return cast(list[dict[str, Any]], self.call(self._kite.get_gtts))

    def delete_gtt(self, trigger_id: int) -> dict[str, Any]:
        """Cancel a GTT by its broker-assigned trigger id."""
        return cast(dict[str, Any], self.call(self._kite.delete_gtt, trigger_id))

    def refresh_access_token(self, request_token: str) -> str:
        """Exchange request token for access token (manual login flow).

        Args:
            request_token: One-time token from Kite redirect after login.

        Returns:
            New access token string.

        Note:
            Requires ``api_secret`` in config. Store the returned token in
            ``KITE_ACCESS_TOKEN`` — tokens expire at end of trading day.
        """
        if not self._config.api_secret:
            msg = "api_secret required for token refresh"
            raise ValueError(msg)
        kite: Any = self._kite
        if not hasattr(kite, "generate_session"):
            msg = "kite instance does not support generate_session"
            raise ValueError(msg)
        session = self.call(
            kite.generate_session,
            request_token,
            api_secret=self._config.api_secret,
        )
        token = str(session["access_token"])
        self._kite.set_access_token(token)
        logger.info("kite_access_token_refreshed")
        return token


def _build_kite(config: KiteConfig) -> KiteConnectProtocol:
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=config.api_key)
    kite.set_access_token(config.access_token or "")
    return cast(KiteConnectProtocol, kite)
