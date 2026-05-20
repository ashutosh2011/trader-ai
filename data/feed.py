"""Abstract bar feed interfaces."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd

from core.bar import Bar


class BarFeed(ABC):
    """Streaming bar source for live or replay modes."""

    @abstractmethod
    def bars(self) -> Iterator[Bar]:
        """Yield bars in chronological order."""

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Materialize all bars as an OHLCV DataFrame."""


class WebsocketFeed(BarFeed):
    """Live websocket bar feed with explicit connect/disconnect lifecycle."""

    @abstractmethod
    async def connect(self) -> None:
        """Open websocket connection and subscribe to instruments."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close websocket connection and release resources."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True when the websocket session is active."""
