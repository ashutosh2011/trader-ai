"""Tests for the hand-rolled retry helper."""

from __future__ import annotations

import pytest

from execution.retry import with_retry


class _Boom(Exception):
    """Retryable boom."""


class _DoNotRetry(Exception):
    """Non-retryable input error."""


def test_succeeds_first_try_no_retry() -> None:
    calls: list[int] = []
    delays: list[float] = []

    def fn() -> str:
        calls.append(1)
        return "ok"

    result = with_retry(
        fn,
        retries=3,
        base_delay=0.01,
        retryable=(_Boom,),
        sleep=delays.append,
    )
    assert result == "ok"
    assert len(calls) == 1
    assert delays == []


def test_fails_twice_succeeds_third_returns_value() -> None:
    attempts: list[int] = []
    delays: list[float] = []

    def fn() -> int:
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise _Boom("transient")
        return 42

    result = with_retry(
        fn,
        retries=3,
        base_delay=0.5,
        max_delay=4.0,
        retryable=(_Boom,),
        sleep=delays.append,
    )
    assert result == 42
    assert len(attempts) == 3
    assert delays == [0.5, 1.0]


def test_always_fails_raises_original_after_retries() -> None:
    delays: list[float] = []
    call_count = 0

    def fn() -> None:
        nonlocal call_count
        call_count += 1
        raise _Boom(f"call_{call_count}")

    with pytest.raises(_Boom, match="call_4"):
        with_retry(
            fn,
            retries=3,
            base_delay=0.1,
            retryable=(_Boom,),
            sleep=delays.append,
        )
    assert call_count == 4
    assert len(delays) == 3


def test_non_retryable_exception_raises_immediately() -> None:
    delays: list[float] = []
    call_count = 0

    def fn() -> None:
        nonlocal call_count
        call_count += 1
        raise _DoNotRetry("bad input")

    with pytest.raises(_DoNotRetry):
        with_retry(
            fn,
            retries=5,
            base_delay=0.1,
            retryable=(_Boom,),
            sleep=delays.append,
        )
    assert call_count == 1
    assert delays == []


def test_max_delay_caps_backoff() -> None:
    delays: list[float] = []
    attempts: list[int] = []

    def fn() -> int:
        attempts.append(0)
        if len(attempts) < 5:
            raise _Boom("retry me")
        return 1

    with_retry(
        fn,
        retries=4,
        base_delay=1.0,
        max_delay=2.5,
        retryable=(_Boom,),
        sleep=delays.append,
    )
    assert delays == [1.0, 2.0, 2.5, 2.5]


def test_invalid_args_rejected() -> None:
    def noop() -> None:
        return None

    with pytest.raises(ValueError):
        with_retry(noop, retries=-1, retryable=(_Boom,))
    with pytest.raises(ValueError):
        with_retry(noop, base_delay=-0.5, retryable=(_Boom,))
    with pytest.raises(ValueError):
        with_retry(noop, base_delay=1.0, max_delay=0.5, retryable=(_Boom,))
