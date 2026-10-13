from __future__ import annotations

import socket
from urllib.error import HTTPError, URLError

import pytest

from aivexa.execution import (
    ErrorKind,
    ExecutionError,
    ExecutionPolicy,
    classify_exception,
    execute_with_retries,
)


def test_slow_success_single_attempt():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        return "ok"

    result, telemetry = execute_with_retries(
        operation,
        ExecutionPolicy(timeout=300, max_attempts=3),
        sleep=lambda _: None,
    )

    assert result == "ok"
    assert calls["n"] == 1
    assert telemetry.attempts == 1
    assert telemetry.final_error is None


def test_transient_failure_is_retried_then_succeeds():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("timed out")
        return "recovered"

    result, telemetry = execute_with_retries(
        operation,
        ExecutionPolicy(timeout=10, max_attempts=3),
        sleep=lambda _: None,
    )

    assert result == "recovered"
    assert calls["n"] == 2
    assert telemetry.attempts == 2


def test_exhausted_retries_raises_classified_error():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        raise TimeoutError("timed out")

    with pytest.raises(ExecutionError) as excinfo:
        execute_with_retries(
            operation,
            ExecutionPolicy(timeout=10, max_attempts=3),
            sleep=lambda _: None,
        )

    assert excinfo.value.kind == ErrorKind.TIMEOUT
    assert calls["n"] == 3


def test_non_retryable_error_is_not_retried():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        raise ExecutionError(
            ErrorKind.MALFORMED,
            "bad payload",
            retryable=False,
        )

    with pytest.raises(ExecutionError):
        execute_with_retries(
            operation,
            ExecutionPolicy(timeout=10, max_attempts=5),
            sleep=lambda _: None,
        )

    assert calls["n"] == 1


def test_classify_timeout_and_connection():
    assert (
        classify_exception(socket.timeout()).kind
        == ErrorKind.TIMEOUT
    )
    assert (
        classify_exception(
            URLError("refused")
        ).kind
        == ErrorKind.CONNECTION
    )


def test_classify_http_status_retryability():
    server_error = HTTPError(
        "http://x", 503, "busy", {}, None
    )
    client_error = HTTPError(
        "http://x", 400, "bad", {}, None
    )

    assert classify_exception(server_error).retryable is True
    assert classify_exception(client_error).retryable is False


def test_policy_validation():
    with pytest.raises(ValueError):
        ExecutionPolicy(max_attempts=0)

    with pytest.raises(ValueError):
        ExecutionPolicy(timeout=0)
