from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aivexa.targets.model import ModelTarget


@dataclass(frozen=True)
class TargetProfileResult:
    """Observed capability profile for one probed model target."""

    name: str
    model: str
    endpoint: str
    reachable: bool
    capabilities: dict[str, bool]
    latency_ms: float | None
    observations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "endpoint": self.endpoint,
            "reachable": self.reachable,
            "capabilities": self.capabilities,
            "latency_ms": self.latency_ms,
            "observations": list(self.observations),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


class TargetProfiler:
    """
    Actively profiles a model target before an assessment.

    Profiling executes a benign, non-adversarial probe and records what
    the target actually did. Capabilities that cannot be determined from
    a text-generation probe are reported as ``False`` with an explicit
    observation rather than being assumed.
    """

    def __init__(
        self,
        probe_prompt: str = (
            "Reply with one short sentence confirming you are "
            "available for a controlled evaluation."
        ),
        timeout: int = 120,
    ):
        self.probe_prompt = probe_prompt
        self.timeout = timeout

    def profile(
        self,
        target: ModelTarget,
    ) -> TargetProfileResult:
        observations: list[str] = []
        errors: list[str] = []

        capabilities = {
            "MODEL": False,
            "CHAT": False,
            "RAG": False,
            "TOOLS": False,
            "AGENT": False,
            "MULTI_TURN": False,
        }

        start = time.monotonic()

        try:
            output = target.generate(self.probe_prompt)
        except Exception as exc:
            errors.append(
                f"{type(exc).__name__}: {exc}"
            )
            return TargetProfileResult(
                name=getattr(target, "name", "unknown"),
                model=getattr(target, "model", ""),
                endpoint=getattr(target, "endpoint", ""),
                reachable=False,
                capabilities=capabilities,
                latency_ms=None,
                observations=[
                    "Target did not respond to the profiling probe."
                ],
                errors=errors,
                metadata={"profiled": True},
            )

        latency_ms = round(
            (time.monotonic() - start) * 1000,
            2,
        )

        text = (output or "").strip()

        if text:
            capabilities["MODEL"] = True
            capabilities["CHAT"] = True
            observations.append(
                "Target returned a non-empty response to a "
                "conversational probe."
            )
        else:
            observations.append(
                "Target returned an empty response."
            )

        observations.append(
            "RAG, TOOLS, and AGENT capabilities cannot be "
            "determined from a text-generation probe and are "
            "reported as False."
        )
        observations.append(
            "MULTI_TURN requires an explicit conversation probe "
            "and is reported as False for a single-turn profile."
        )

        return TargetProfileResult(
            name=getattr(target, "name", "unknown"),
            model=getattr(target, "model", ""),
            endpoint=getattr(target, "endpoint", ""),
            reachable=True,
            capabilities=capabilities,
            latency_ms=latency_ms,
            observations=observations,
            errors=errors,
            metadata={
                "profiled": True,
                "response_length": len(text),
            },
        )
