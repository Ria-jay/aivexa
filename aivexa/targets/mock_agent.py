from __future__ import annotations

from typing import Any

from aivexa.evidence.system import SystemInteraction
from aivexa.targets.application import AIApplicationTarget


class MockAgentTarget(AIApplicationTarget):
    """
    Deterministic local agent/tool-using target for controlled tests.

    It is safe by construction: it performs no real actions. Instead it
    returns scripted, observable tool calls, tool results, and agent
    traces so the AIVEXA agent/tool-use evaluators can be exercised
    end-to-end without touching a real service.

    A ``scenarios`` list allows response selection by input substring, so
    one test double can model both safe and unsafe behavior.
    """

    def __init__(
        self,
        name: str = "aivexa-mock-agent",
        endpoint: str = "local://mock-agent",
        *,
        system_output: str = "Agent task completed.",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        agent_trace: list[dict[str, Any]] | None = None,
        retrieved_context: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        scenarios: list[dict[str, Any]] | None = None,
    ):
        super().__init__(name=name, endpoint=endpoint)

        self.defaults = {
            "system_output": system_output,
            "tool_calls": list(tool_calls or []),
            "tool_results": list(tool_results or []),
            "agent_trace": list(agent_trace or []),
            "retrieved_context": list(retrieved_context or []),
            "context": dict(context or {}),
            "metadata": dict(metadata or {}),
        }

        self.scenarios = list(scenarios or [])

    def _select(self, user_input: str) -> dict[str, Any]:
        for scenario in self.scenarios:
            match = scenario.get("match", "")

            if match and match in user_input:
                return {
                    **self.defaults,
                    **scenario,
                }

        return dict(self.defaults)

    def interact(
        self,
        experiment_id: str,
        user_input: str,
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
    ) -> SystemInteraction:
        selected = self._select(user_input)

        metadata = {
            "target_type": "ai_application",
            "implementation": "mock_agent",
            "target_name": self.name,
            "endpoint": self.endpoint,
            **selected.get("metadata", {}),
        }

        if selected.get("tool_results"):
            metadata["tool_results"] = list(
                selected["tool_results"]
            )

        if selected.get("agent_trace"):
            metadata["agent_trace"] = list(
                selected["agent_trace"]
            )

        return SystemInteraction(
            experiment_id=experiment_id,
            user_input=user_input,
            system_output=str(
                selected.get("system_output", "")
            ),
            context={
                **selected.get("context", {}),
                **dict(context or {}),
            },
            retrieved_context=list(
                selected.get("retrieved_context", [])
            ),
            tool_calls=list(selected.get("tool_calls", [])),
            authorization_context=dict(
                authorization_context or {}
            ),
            observations=[
                "Interaction completed by controlled mock agent."
            ],
            metadata=metadata,
        )
