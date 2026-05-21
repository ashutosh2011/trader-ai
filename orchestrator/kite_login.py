"""Zerodha Kite Connect login helper — obtain and persist daily access token."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from kiteconnect import KiteConnect

DEFAULT_ENV_PATH = Path(".env")


def kite_login_url(api_key: str) -> str:
    """Return the Kite login URL for the user to authenticate in a browser."""
    return cast(str, KiteConnect(api_key=api_key).login_url())


def exchange_request_token(
    api_key: str,
    api_secret: str,
    request_token: str,
) -> str:
    """Exchange a one-time request token for a session access token."""
    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    return str(session["access_token"])


def update_env_access_token(
    access_token: str,
    env_path: Path = DEFAULT_ENV_PATH,
) -> None:
    """Set or replace KITE_ACCESS_TOKEN in a .env file."""
    if not env_path.is_file():
        msg = f".env not found at {env_path.resolve()}"
        raise FileNotFoundError(msg)

    text = env_path.read_text(encoding="utf-8")
    line = f"KITE_ACCESS_TOKEN={access_token}\n"
    if re.search(r"^KITE_ACCESS_TOKEN=.*$", text, flags=re.MULTILINE):
        text = re.sub(
            r"^KITE_ACCESS_TOKEN=.*$",
            line.rstrip("\n"),
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += line
    env_path.write_text(text, encoding="utf-8")


def run_interactive_login(
    api_key: str,
    api_secret: str,
    *,
    request_token: str | None = None,
    env_path: Path = DEFAULT_ENV_PATH,
    save: bool = True,
) -> str:
    """Print login URL, exchange request token, optionally persist to .env."""
    click_url = kite_login_url(api_key)
    print("Open this URL in your browser and log in to Zerodha:")
    print(click_url)
    print()
    print("After login, copy the request_token from the redirect URL query string.")

    token = request_token
    if not token:
        token = input("request_token: ").strip()
    if not token:
        msg = "request_token is required"
        raise ValueError(msg)

    access_token = exchange_request_token(api_key, api_secret, token)
    if save:
        update_env_access_token(access_token, env_path)
        print(f"Saved KITE_ACCESS_TOKEN to {env_path.resolve()}")
    else:
        print("Add to .env:")
        print(f"KITE_ACCESS_TOKEN={access_token}")
    print()
    print("Note: access tokens expire at end of each trading day. Re-run kite-login daily.")
    return access_token
