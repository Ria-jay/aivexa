from typing import Any

from aivexa.evidence.system import SystemInteraction
from aivexa.targets.application import AIApplicationTarget


class MockAIApplicationTarget(AIApplicationTarget):
    """
    Deterministic local application target for Phase 4 development.

    The optional interaction fields allow permanent tests to model
    controlled application behavior without changing the production
    HTTP target implementation.
    """

    def __init__(
        self,
        name: str = "aivexa-mock-application",
        endpoint: str = "local://mock",
        retrieved_context: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        observations: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(
            name=name,
            endpoint=endpoint,
        )

        self.retrieved_context = list(
            retrieved_context or []
        )
        self.tool_calls = list(
            tool_calls or []
        )
        self.observations = list(
            observations or [
                "Interaction completed by controlled mock application."
            ]
        )
        self.metadata = dict(
            metadata or {}
        )

    def interact(
        self,
        experiment_id: str,
        user_input: str,
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
    ) -> SystemInteraction:
        context = dict(context or {})
        authorization_context = dict(
            authorization_context or {}
        )

        metadata = {
            "target_type": "ai_application",
            "implementation": "mock",
            "target_name": self.name,
            "endpoint": self.endpoint,
            **self.metadata,
        }

        return SystemInteraction(
            experiment_id=experiment_id,
            user_input=user_input,
            system_output=(
                f"Mock application response to: {user_input}"
            ),
            context=context,
            retrieved_context=list(
                self.retrieved_context
            ),
            tool_calls=list(
                self.tool_calls
            ),
            authorization_context=authorization_context,
            observations=list(
                self.observations
            ),
            metadata=metadata,
        )
