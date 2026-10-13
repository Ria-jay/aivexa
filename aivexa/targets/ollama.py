import json
from urllib import request
from urllib.error import HTTPError, URLError

from aivexa.execution import (
    ErrorKind,
    ExecutionError,
    ExecutionPolicy,
    execute_with_retries,
)


class OllamaTarget:
    """
    Ollama model adapter using AIVEXA's shared execution policy.

    A single request may be retried for transient failures only; the
    per-attempt socket timeout and the retry/backoff settings are
    configurable through an ``ExecutionPolicy``.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: int = 120,
        policy: ExecutionPolicy | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.policy = policy or ExecutionPolicy(timeout=timeout)
        self.timeout = self.policy.timeout
        self.last_telemetry = None

    def _request(self, prompt: str, options: dict | None) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": options or {},
            }
        ).encode("utf-8")

        req = request.Request(
            f"{self.endpoint}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(
                req,
                timeout=self.policy.timeout,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )
        except (HTTPError, URLError):
            raise
        except json.JSONDecodeError as exc:
            raise ExecutionError(
                ErrorKind.MALFORMED,
                "Ollama returned malformed JSON",
                retryable=False,
            ) from exc

        if "response" not in data:
            raise ExecutionError(
                ErrorKind.MALFORMED,
                "Ollama response is missing the 'response' field",
                retryable=False,
            )

        return data["response"]

    def generate(self, prompt: str, options: dict | None = None) -> str:
        result, telemetry = execute_with_retries(
            lambda: self._request(prompt, options),
            self.policy,
        )

        self.last_telemetry = telemetry
        return result
