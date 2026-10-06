from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from aivexa.evaluation.adversarial import AdversarialCase
from aivexa.evaluation.models import Evaluation, ExperimentResult


@dataclass(frozen=True)
class AdversarialObservation:
    variant_id: str
    baseline_output: str
    variant_output: str
    context: dict[str, Any]
    observations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "baseline_output": self.baseline_output,
            "variant_output": self.variant_output,
            "context": self.context,
            "observations": self.observations,
        }


class AdversarialEvaluationEngine:
    """
    Property-driven adversarial evaluation.

    The engine does not define "jailbreak" as a universal failure.
    Instead, it evaluates whether the intervention affected the
    security/safety property declared by the test case.

    A property evaluator can be supplied for application-specific
    semantics. Without one, the engine records the behavioral
    comparison as INCONCLUSIVE rather than pretending to prove a
    security violation.
    """

    evaluator_name = "aivexa-adversarial-engine"

    def evaluate(
        self,
        case: AdversarialCase,
        observations: list[AdversarialObservation],
        property_evaluator: Callable[
            [AdversarialCase, list[AdversarialObservation]],
            Evaluation,
        ]
        | None = None,
    ) -> Evaluation:
        if not observations:
            return Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "No executable observations were available for "
                    "the adversarial evaluation."
                ),
                confidence="Low",
                property_name=case.property_name,
                property_expectation=case.property_expectation,
                evaluator=self.evaluator_name,
                evidence={
                    "observation_count": 0,
                },
            )

        if property_evaluator is not None:
            return property_evaluator(
                case,
                observations,
            )

        changed = sum(
            1
            for observation in observations
            if observation.baseline_output
            != observation.variant_output
        )

        return Evaluation(
            result=ExperimentResult.INCONCLUSIVE,
            rationale=(
                "The adversarial intervention produced observable "
                "behavioral evidence, but no property-specific "
                "evaluator was supplied to establish a security result."
            ),
            confidence="Low",
            property_name=case.property_name,
            property_expectation=case.property_expectation,
            evaluator=self.evaluator_name,
            evidence={
                "observation_count": len(observations),
                "changed_observations": changed,
                "unchanged_observations": (
                    len(observations) - changed
                ),
                "observations": [
                    observation.to_dict()
                    for observation in observations
                ],
            },
        )
