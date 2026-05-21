"""Hand-rolled exponential backoff retry helper for sync broker calls.

TRADEOFF: We deliberately do not pull in ``tenacity`` because it is not in the
pinned dependency stack and the surface we need is tiny (sync only, fixed
exception tuple, log per attempt). The synchronous shape matches the
kiteconnect SDK which is itself sync.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import structlog

T = TypeVar("T")

logger = structlog.get_logger(__name__)


def with_retry(
    fn: Callable[[], T],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    retryable: tuple[type[BaseException], ...],
    logger_: structlog.stdlib.BoundLogger | None = None,
    op: str = "operation",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Invoke ``fn`` with exponential backoff retry on ``retryable`` errors.

    The total number of invocations is at most ``retries + 1``: one initial
    attempt plus up to ``retries`` re-tries. Each retry waits
    ``min(max_delay, base_delay * 2**attempt)`` seconds before the next call
    (``attempt`` is zero-indexed at the *first* retry).

    Args:
        fn: The synchronous, side-effecting callable to invoke. It receives no
            arguments — wrap with ``functools.partial`` or a lambda if you need
            to bind parameters.
        retries: Maximum number of *additional* attempts after the first
            failure. ``retries=0`` disables retry entirely.
        base_delay: Initial backoff in seconds. Must be ``>= 0``.
        max_delay: Hard cap on per-attempt backoff. Must be ``>= base_delay``.
        retryable: Tuple of exception types that should trigger a retry. Any
            other exception type is re-raised immediately (fail-fast for bad
            input like ``InputException`` and ``OrderException``).
        logger_: Optional pre-bound structlog logger; defaults to the module
            logger.
        op: Short label included in every log line for context.
        sleep: Override for ``time.sleep`` — primarily for tests so they can
            assert backoff scheduling without waiting in real time.

    Returns:
        The return value of ``fn`` on success.

    Raises:
        BaseException: The last exception observed once retries are exhausted,
            or the original exception when a non-retryable type is seen.
    """
    if retries < 0:
        msg = f"retries must be >= 0 (got {retries})"
        raise ValueError(msg)
    if base_delay < 0:
        msg = f"base_delay must be >= 0 (got {base_delay})"
        raise ValueError(msg)
    if max_delay < base_delay:
        msg = f"max_delay ({max_delay}) must be >= base_delay ({base_delay})"
        raise ValueError(msg)

    log = logger_ or logger
    attempt = 0
    while True:
        try:
            return fn()
        except retryable as exc:
            if attempt >= retries:
                log.warning(
                    "retry_exhausted",
                    op=op,
                    attempt=attempt,
                    retries=retries,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )
                raise
            delay = min(max_delay, base_delay * (2**attempt))
            log.warning(
                "retry_attempt",
                op=op,
                attempt=attempt,
                retries=retries,
                delay=delay,
                error=str(exc),
                exc_type=type(exc).__name__,
            )
            sleep(delay)
            attempt += 1
