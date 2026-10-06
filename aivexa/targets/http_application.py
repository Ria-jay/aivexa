import json
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from aivexa.evidence.system import SystemInteraction
from aivexa.targets.application import AIApplicationTarget


class HTTPAIApplicationTarget(AIApplicationTarget):
    """
    Generic HTTP target adapter for AI-powered applications.

    The adapter intentionally does not assume a vendor-specific API.
    Request and response mapping are configurable so the AIVEXA
    evaluation layer remains independent of the target implementation.
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        request_builder=None,
        response_parser=None,
        headers: dict[str, str] | None = None,
        timeout: int = 120,
    ):
        super().__init__(name=name, endpoint=endpoint)
        self.request_builder = (
            request_builder or self._default_request_builder
        )
        self.response_parser = (
            response_parser or self._default_response_parser
        )
        self.headers = headers or {
            "Content-Type": "application/json"
        }
        self.timeout = timeout

    def interact(
        self,
        experiment_id: str,
        user_input: str,
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
    ) -> SystemInteraction:
        context = context or {}
        authorization_context = authorization_context or {}

        payload = self.request_builder(
            experiment_id=experiment_id,
            user_input=user_input,
            context=context,
            authorization_context=authorization_context,
        )

        body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            self.endpoint,
            data=body,
            headers=self.headers,
            method="POST",
        )

        try:
            with request.urlopen(
                req,
                timeout=self.timeout,
            ) as response:
                raw_response = response.read().decode("utf-8")
                status_code = response.status
        except HTTPError as exc:
            raise RuntimeError(
                f"AI application returned HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to AI application at "
                f"{self.endpoint}"
            ) from exc

        parsed = self.response_parser(raw_response)

        return SystemInteraction(
            experiment_id=experiment_id,
            user_input=user_input,
            system_output=parsed["system_output"],
            context=context,
            retrieved_context=parsed.get(
                "retrieved_context",
                [],
            ),
            tool_calls=parsed.get(
                "tool_calls",
                [],
            ),
            authorization_context=authorization_context,
            observations=parsed.get(
                "observations",
                [],
            ),
            metadata={
                **parsed.get("metadata", {}),
                "target_type": "ai_application",
                "implementation": "http",
                "target_name": self.name,
                "endpoint": self.endpoint,
                "http_status": status_code,
            },
        )

    @staticmethod
    def _default_request_builder(
        experiment_id: str,
        user_input: str,
        context: dict[str, Any],
        authorization_context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "user_input": user_input,
            "context": context,
            "authorization_context": authorization_context,
        }

    @staticmethod
    def _default_response_parser(
        raw_response: str,
    ) -> dict[str, Any]:
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "AI application returned invalid JSON"
            ) from exc

        if "system_output" not in data:
            raise RuntimeError(
                "AI application response is missing "
                "'system_output'"
            )

        return data
