from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError


T = TypeVar("T")


class ErrorKind(str, Enum):
    CONNECTION = "CONNECTION"
    TIMEOUT = "TIMEOUT"
    HTTP_STATUS = "HTTP_STATUS"
    MALFORMED = "MALFORMED"
    UNKNOWN = "UNKNOWN"


class ExecutionError(RuntimeError):
    """
    Classified execution failure.

    ``retryable`` is True only for transient failures (connection drops,
    timeouts, and 429/5xx responses). Deterministic failures such as a
    4xx client error or a malformed response are not retried.
    """

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
    ):
        self.kind = kind
        self.status = status
        self.retryable = retryable
        super().__init__(message)


def classify_exception(exc: Exception) -> ExecutionError:
    """Map a transport exception to a classified ExecutionError."""
    if isinstance(exc, ExecutionError):
        return exc

    if isinstance(exc, HTTPError):
        status = exc.code
        retryable = status == 429 or status >= 500
        return ExecutionError(
            ErrorKind.HTTP_STATUS,
            f"HTTP {status}",
            status=status,
            retryable=retryable,
        )

    if isinstance(exc, URLError):
        return ExecutionError(
            ErrorKind.CONNECTION,
            f"Connection error: {exc.reason}",
            retryable=True,
        )

    if isinstance(exc, (socket.timeout, TimeoutError)):
        return ExecutionError(
            ErrorKind.TIMEOUT,
            "Request timed out",
            retryable=True,
        )

    return ExecutionError(
        ErrorKind.UNKNOWN,
        f"{type(exc).__name__}: {exc}",
        retryable=False,
    )


@dataclass(frozen=True)
class ExecutionPolicy:
    """
    Provider-aware execution policy shared by all model adapters.

    ``timeout`` is the per-attempt socket timeout (connection + read).
    ``max_attempts`` bounds retries for transient failures only.
    ``overall_deadline`` caps total wall-clock time across attempts and
    backoff; when exceeded, no further attempts are made.
    """

    timeout: float = 300.0
    max_attempts: int = 3
    backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 30.0
    overall_deadline: float | None = 900.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")


@dataclass(frozen=True)
class AttemptTelemetry:
    attempts: int
    elapsed_seconds: float
    final_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "final_error": self.final_error,
        }


def execute_with_retries(
    operation: Callable[[], T],
    policy: ExecutionPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, AttemptTelemetry]:
    """
    Run ``operation`` under a bounded retry policy.

    Retries only transient, classified failures. Respects the overall
    deadline. Raises the final ExecutionError when all attempts fail.
    """
    start = time.monotonic()
    delay = policy.backoff_seconds
    attempts = 0
    last_error: ExecutionError | None = None

    while attempts < policy.max_attempts:
        attempts += 1

        try:
            result = operation()
            return result, AttemptTelemetry(
                attempts=attempts,
                elapsed_seconds=time.monotonic() - start,
                final_error=None,
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            error = classify_exception(exc)
            last_error = error

            if not error.retryable:
                raise error

            if attempts >= policy.max_attempts:
                break

            if policy.overall_deadline is not None:
                elapsed = time.monotonic() - start
                if elapsed + delay > policy.overall_deadline:
                    break

            sleep(delay)
            delay = min(
                delay * policy.backoff_multiplier,
                policy.max_backoff_seconds,
            )

    assert last_error is not None
    raise last_error
