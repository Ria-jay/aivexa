from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_string(value: str, input_value: str) -> str:
    value = value.replace("${input}", input_value)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "input":
            return input_value

        resolved = os.environ.get(name)
        if resolved is None:
            raise RuntimeError(
                f"Required environment variable '{name}' is not set."
            )
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def resolve_templates(value: Any, input_value: str) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, input_value)

    if isinstance(value, list):
        return [
            resolve_templates(item, input_value)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: resolve_templates(item, input_value)
            for key, item in value.items()
        }

    return value


def extract_path(data: Any, path: str | None) -> Any:
    if not path:
        return None

    current = data

    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue

        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
            continue

        return None

    return current


@dataclass(frozen=True)
class TargetProfile:
    name: str
    endpoint: str
    authorized: bool
    authorization_note: str
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    request_body: dict[str, Any] = field(default_factory=dict)
    response_paths: dict[str, str] = field(default_factory=dict)
    timeout: int = 60

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetProfile":
        return cls(
            name=str(data["name"]),
            endpoint=str(data["endpoint"]),
            authorized=bool(data.get("authorized", False)),
            authorization_note=str(
                data.get("authorization_note", "")
            ),
            method=str(data.get("method", "POST")).upper(),
            headers=dict(data.get("headers", {})),
            request_body=dict(data.get("request_body", {})),
            response_paths=dict(data.get("response_paths", {})),
            timeout=int(data.get("timeout", 60)),
        )

    @classmethod
    def load(cls, path: str) -> "TargetProfile":
        profile_path = Path(path)

        if not profile_path.exists():
            raise FileNotFoundError(
                f"Target profile not found: {path}"
            )

        data = json.loads(
            profile_path.read_text(encoding="utf-8")
        )

        profile = cls.from_dict(data)
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Target profile name cannot be empty.")

        if not self.endpoint.strip():
            raise ValueError("Target endpoint cannot be empty.")

        if self.method not in {
            "GET",
            "POST",
            "PUT",
            "PATCH",
        }:
            raise ValueError(
                f"Unsupported HTTP method: {self.method}"
            )

        if not self.authorization_note.strip():
            raise ValueError(
                "Target profile must contain authorization_note."
            )

        if "input" not in str(self.request_body):
            raise ValueError(
                "request_body must contain ${input} so AIVEXA "
                "can control the tested input."
            )

    def request_headers(self, input_value: str) -> dict[str, str]:
        resolved = resolve_templates(
            self.headers,
            input_value,
        )

        return {
            str(key): str(value)
            for key, value in resolved.items()
        }

    def request_payload(self, input_value: str) -> dict[str, Any]:
        return resolve_templates(
            self.request_body,
            input_value,
        )
