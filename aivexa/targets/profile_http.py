from __future__ import annotations

import json
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from aivexa.evidence.system import SystemInteraction
from aivexa.targets.application import AIApplicationTarget
from aivexa.targets.profile import (
    TargetProfile,
    extract_path,
)


class ProfileHTTPApplicationTarget(AIApplicationTarget):
    """
    Generic authorized HTTP AI-application target driven by a
    TargetProfile.

    No vendor-specific behavior is embedded here.
    """

    def __init__(self, profile: TargetProfile):
        profile.validate()

        if not profile.authorized:
            raise PermissionError(
                "Target profile is not authorized. "
                "Set authorized=true only for a target you are "
                "explicitly permitted to assess."
            )

        super().__init__(
            name=profile.name,
            endpoint=profile.endpoint,
        )

        self.profile = profile

    def interact(
        self,
        experiment_id: str,
        user_input: str,
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
    ) -> SystemInteraction:
        headers = self.profile.request_headers(user_input)
        payload = self.profile.request_payload(user_input)

        body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            self.profile.endpoint,
            data=body,
            headers=headers,
            method=self.profile.method,
        )

        status_code: int | None = None
        response_headers: dict[str, str] = {}

        try:
            with request.urlopen(
                req,
                timeout=self.profile.timeout,
            ) as response:
                status_code = response.status
                response_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() not in {
                        "authorization",
                        "cookie",
                        "set-cookie",
                    }
                }

                raw = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

        except HTTPError as exc:
            status_code = exc.code
            raise RuntimeError(
                f"Target returned HTTP {exc.code}"
            ) from exc

        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to target at "
                f"{self.profile.endpoint}"
            ) from exc

        try:
            response_data: Any = json.loads(raw)
        except json.JSONDecodeError:
            response_data = raw

        output_path = self.profile.response_paths.get(
            "system_output"
        )

        system_output = extract_path(
            response_data,
            output_path,
        )

        if system_output is None:
            if isinstance(response_data, str):
                system_output = response_data
            else:
                system_output = json.dumps(
                    response_data,
                    ensure_ascii=False,
                )

        retrieved_context = extract_path(
            response_data,
            self.profile.response_paths.get(
                "retrieved_context"
            ),
        )

        tool_calls = extract_path(
            response_data,
            self.profile.response_paths.get(
                "tool_calls"
            ),
        )

        observations = extract_path(
            response_data,
            self.profile.response_paths.get(
                "observations"
            ),
        )

        metadata = extract_path(
            response_data,
            self.profile.response_paths.get(
                "metadata"
            ),
        )

        if not isinstance(retrieved_context, list):
            retrieved_context = []

        if not isinstance(tool_calls, list):
            tool_calls = []

        if not isinstance(observations, list):
            observations = []

        if not isinstance(metadata, dict):
            metadata = {}

        metadata = {
            **metadata,
            "http": {
                "status_code": status_code,
                "response_headers": response_headers,
            },
            "target_profile": self.profile.name,
        }

        return SystemInteraction(
            experiment_id=experiment_id,
            user_input=user_input,
            system_output=str(system_output),
            context=dict(context or {}),
            retrieved_context=retrieved_context,
            tool_calls=tool_calls,
            authorization_context=dict(
                authorization_context or {}
            ),
            observations=[
                str(item)
                for item in observations
            ],
            metadata=metadata,
        )
