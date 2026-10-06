from __future__ import annotations

from typing import Any

from aivexa.evidence.models import Evidence
from aivexa.targets.model import ModelTarget


class ExperimentRunner:
    """
    Executes model experiments through the provider-independent
    ModelTarget interface.
    """

    def __init__(self, target: ModelTarget):
        self.target = target

    def run(
        self,
        experiment_id: str,
        prompt: str,
        generation_options: dict[str, Any] | None = None,
    ) -> Evidence:
        response = self.target.generate(
            prompt,
            options=generation_options,
        )

        return Evidence(
            experiment_id=experiment_id,
            input_data=prompt,
            output_data=response,
            observations=[],
            metadata={
                "target_type": "model",
                "provider": type(self.target).__name__,
                "model": self.target.model,
                "endpoint": self.target.endpoint,
                "generation_options": (
                    generation_options or {}
                ),
            },
        )
