from __future__ import annotations

from typing import Any, Protocol


class ModelTarget(Protocol):
    """
    Minimal provider-independent interface required by AIVEXA's
    model experiment runner.

    Ollama, Gemini, and future providers can implement this interface
    without changing the experiment or evaluation layers.
    """

    name: str
    model: str
    endpoint: str

    def generate(
        self,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        ...
