from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aivexa.evaluation.adversarial import (
    AdversarialCase,
    TargetCapability,
)


@dataclass(frozen=True)
class AdversarialTargetProfile:
    target_name: str
    capabilities: frozenset[TargetCapability]
    authorized: bool = False
    authorization_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(
        self,
        capability: TargetCapability,
    ) -> bool:
        return capability in self.capabilities

    def applicable(
        self,
        case: AdversarialCase,
    ) -> bool:
        return case.applicable_to(
            set(self.capabilities)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "capabilities": [
                capability.value
                for capability in sorted(
                    self.capabilities,
                    key=lambda item: item.value,
                )
            ],
            "authorized": self.authorized,
            "authorization_note": self.authorization_note,
            "metadata": self.metadata,
        }


class AdversarialCaseSelector:
    def select(
        self,
        cases: tuple[AdversarialCase, ...],
        profile: AdversarialTargetProfile,
    ) -> tuple[AdversarialCase, ...]:
        if not profile.authorized:
            raise PermissionError(
                "Adversarial evaluation requires an authorized target."
            )

        return tuple(
            case
            for case in cases
            if profile.applicable(case)
        )

    def applicability(
        self,
        cases: tuple[AdversarialCase, ...],
        profile: AdversarialTargetProfile,
    ) -> list[dict[str, Any]]:
        return [
            {
                "case_id": case.case_id,
                "applicable": profile.applicable(case),
                "required_capabilities": [
                    capability.value
                    for capability in case.required_capabilities
                ],
            }
            for case in cases
        ]
