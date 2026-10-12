from __future__ import annotations

from typing import Any

from aivexa.targets.config import TargetConfig
from aivexa.targets.ollama import OllamaTarget
from aivexa.targets.openai_compatible import (
    OpenAICompatibleTarget,
)


class UnsupportedTargetTypeError(ValueError):
    """Raised when a target type has no implemented adapter."""


# Target types that provide a ModelTarget-compatible adapter.
MODEL_TARGET_TYPES = frozenset(
    {
        "ollama",
        "openai",
        "openai-compatible",
        "openai_compatible",
    }
)


def build_model_target(
    config: TargetConfig,
) -> Any:
    """
    Create a ``ModelTarget`` from a validated ``TargetConfig``.

    Only provider types that have a real, exercised adapter are
    supported. Unknown types are rejected explicitly rather than
    silently producing a non-functional target.
    """
    config.validate()

    target_type = config.target_type.strip().lower()

    if target_type == "ollama":
        return OllamaTarget(
            endpoint=config.endpoint,
            model=config.model,
        )

    if target_type in {
        "openai",
        "openai-compatible",
        "openai_compatible",
    }:
        return OpenAICompatibleTarget(
            name=config.name,
            model=config.model,
            endpoint=config.endpoint,
            api_key_env=config.api_key_env,
            timeout=config.timeout,
        )

    raise UnsupportedTargetTypeError(
        f"Target type '{config.target_type}' does not have an "
        "implemented model adapter. Supported model target types: "
        + ", ".join(sorted(MODEL_TARGET_TYPES))
    )


def check_prerequisites(config: TargetConfig) -> list[str]:
    """
    Return a list of human-readable prerequisite problems for a target
    configuration. An empty list means the configuration is usable.
    """
    problems: list[str] = []

    try:
        config.validate()
    except ValueError as exc:
        problems.append(str(exc))
        return problems

    target_type = config.target_type.strip().lower()

    if target_type not in MODEL_TARGET_TYPES:
        problems.append(
            f"No model adapter is implemented for target type "
            f"'{config.target_type}'."
        )

    if config.api_key_env:
        import os

        if not os.environ.get(config.api_key_env):
            problems.append(
                f"Environment variable '{config.api_key_env}' is "
                "not set; the target may reject unauthenticated "
                "requests."
            )

    return problems
