from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from aivexa.evidence.system import SystemInteraction


# Step types for a multi-step agent trace.
MODEL_OUTPUT = "MODEL_OUTPUT"
TOOL_CALL = "TOOL_CALL"
TOOL_RESULT = "TOOL_RESULT"
FINAL = "FINAL"


@dataclass(frozen=True)
class AgentStep:
    """One observable step in an agent execution trace."""

    index: int
    step_type: str
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    instruction_source: str | None = None
    trusted: bool | None = None
    authorized: bool | None = None
    confirmed: bool | None = None
    result: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentTrace:
    """Ordered sequence of observable agent steps."""

    experiment_id: str
    steps: tuple[AgentStep, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "steps": [step.to_dict() for step in self.steps],
        }

    @property
    def tool_calls(self) -> tuple[AgentStep, ...]:
        return tuple(
            step
            for step in self.steps
            if step.step_type == TOOL_CALL
        )

    @property
    def tool_results(self) -> tuple[AgentStep, ...]:
        return tuple(
            step
            for step in self.steps
            if step.step_type == TOOL_RESULT
        )


def _step_from_dict(
    index: int,
    data: dict[str, Any],
) -> AgentStep:
    return AgentStep(
        index=index,
        step_type=str(data.get("step_type", TOOL_CALL)),
        tool=data.get("tool"),
        arguments=dict(data.get("arguments", {})),
        instruction_source=data.get("instruction_source"),
        trusted=data.get("trusted"),
        authorized=data.get("authorized"),
        confirmed=data.get("confirmed"),
        result=dict(data.get("result", {})),
        metadata=dict(data.get("metadata", {})),
    )


def build_trace(interaction: SystemInteraction) -> AgentTrace:
    """
    Build an agent trace from an observable system interaction.

    An explicit ``metadata["agent_trace"]`` list takes precedence. When
    it is absent, a trace is derived from the observed ``tool_calls``
    and ``metadata["tool_results"]`` so existing targets work without
    change.
    """
    explicit = interaction.metadata.get("agent_trace")

    if isinstance(explicit, list) and explicit:
        steps = [
            _step_from_dict(index, data)
            for index, data in enumerate(explicit)
            if isinstance(data, dict)
        ]

        return AgentTrace(
            experiment_id=interaction.experiment_id,
            steps=tuple(steps),
        )

    steps: list[AgentStep] = []
    index = 0

    for call in interaction.tool_calls:
        if not isinstance(call, dict):
            continue

        steps.append(
            AgentStep(
                index=index,
                step_type=TOOL_CALL,
                tool=call.get("tool"),
                arguments=dict(call.get("arguments", {})),
                instruction_source=call.get(
                    "instruction_source"
                ),
                trusted=call.get("trusted"),
                authorized=call.get("authorized"),
                confirmed=call.get("confirmed"),
                metadata=dict(call.get("metadata", {})),
            )
        )
        index += 1

    for result in interaction.metadata.get("tool_results", []):
        if not isinstance(result, dict):
            continue

        steps.append(
            AgentStep(
                index=index,
                step_type=TOOL_RESULT,
                tool=result.get("tool"),
                trusted=result.get("trusted"),
                result=dict(result),
                metadata=dict(result.get("metadata", {})),
            )
        )
        index += 1

    return AgentTrace(
        experiment_id=interaction.experiment_id,
        steps=tuple(steps),
    )
