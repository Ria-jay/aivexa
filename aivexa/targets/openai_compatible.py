from __future__ import annotations

import json
import os
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from aivexa.execution import (
    ErrorKind,
    ExecutionError,
    ExecutionPolicy,
    execute_with_retries,
)


class OpenAICompatibleTarget:
    """
    Provider-independent adapter for OpenAI-compatible chat completion
    endpoints.

    It implements the AIVEXA ``ModelTarget`` interface directly, so model
    safety and model security evaluations can run against any
    OpenAI-compatible deployment (OpenAI, Azure-compatible gateways,
    vLLM, llama.cpp server, LM Studio, etc.) without changing the
    evaluation layer.

    Credentials are never stored in code; the API key is read from the
    configured environment variable at construction time. Transient
    failures are retried under AIVEXA's shared execution policy.
    """

    def __init__(
        self,
        name: str,
        model: str,
        endpoint: str,
        api_key_env: str = "",
        timeout: int = 120,
        policy: ExecutionPolicy | None = None,
    ):
        if not endpoint.strip():
            raise ValueError(
                "OpenAI-compatible target requires an endpoint."
            )

        if not model.strip():
            raise ValueError(
                "OpenAI-compatible target requires a model."
            )

        self.name = name
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.api_key_env = api_key_env
        self.policy = policy or ExecutionPolicy(timeout=timeout)
        self.timeout = self.policy.timeout
        self.last_telemetry = None
        self.api_key = (
            os.environ.get(api_key_env)
            if api_key_env
            else None
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        return headers

    def _request(
        self,
        prompt: str,
        options: dict[str, Any] | None,
    ) -> str:
        options = dict(options or {})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if "temperature" in options:
            payload["temperature"] = options["temperature"]

        if "num_predict" in options:
            payload["max_tokens"] = options["num_predict"]

        body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            f"{self.endpoint}/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )

        try:
            with request.urlopen(
                req,
                timeout=self.policy.timeout,
            ) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(
                ErrorKind.UNKNOWN,
                f"{type(exc).__name__}: {exc}",
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExecutionError(
                ErrorKind.MALFORMED,
                "OpenAI-compatible endpoint returned invalid JSON",
                retryable=False,
            ) from exc

        choices = data.get("choices")

        if not choices:
            raise ExecutionError(
                ErrorKind.MALFORMED,
                "OpenAI-compatible endpoint returned no choices.",
                retryable=False,
            )

        message = choices[0].get("message", {})

        content = message.get("content")

        if content is None:
            raise ExecutionError(
                ErrorKind.MALFORMED,
                "OpenAI-compatible response contained no message content.",
                retryable=False,
            )

        return str(content)

    def generate(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        result, telemetry = execute_with_retries(
            lambda: self._request(prompt, options),
            self.policy,
        )

        self.last_telemetry = telemetry
        return result
