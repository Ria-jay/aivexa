from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class SystemInteraction:
    """
    Observable interaction with an AI-powered application or system.

    This model records what AIVEXA can observe during an experiment.
    It does not assume that an observed behavior is a vulnerability.
    """

    experiment_id: str

    user_input: str

    system_output: str

    context: dict[str, Any] = field(default_factory=dict)

    retrieved_context: list[dict[str, Any]] = field(default_factory=list)

    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    authorization_context: dict[str, Any] = field(default_factory=dict)

    observations: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    observed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True)
class SystemObservation:
    """
    Structured observation extracted from a system interaction.

    Observations describe behavior. They are deliberately separate from
    findings and vulnerabilities.
    """

    name: str

    value: Any

    source: str

    confidence: str = "Medium"

    metadata: dict[str, Any] = field(default_factory=dict)
