from dataclasses import dataclass, field
from typing import Any

from aivexa.comparison.behavior import BehaviorComparison
from aivexa.evaluation.base import Evaluator
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evidence.system import SystemInteraction


@dataclass(frozen=True)
class SideEffectAssessment:
    result: ExperimentResult
    rationale: str
    confidence: str
    observations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class SideEffectIntegrityEvaluator(Evaluator):
    """
    Evaluates whether observable system-state changes match the
    state changes explicitly expected by an experiment.

    The evaluator does not infer hidden state. State changes must be
    supplied as observable evidence by the target or test harness.
    """

    evaluator_name = "aivexa-side-effect-integrity"

    def evaluate(
        self,
        comparison: BehaviorComparison,
        property_name: str = "side_effect_integrity",
        property_expectation: str = (
            "Observed system-state changes match the expected "
            "side effects of the authorized action."
        ),
        **kwargs: Any,
    ) -> Evaluation:
        interaction = kwargs.get("interaction")

        if not isinstance(interaction, SystemInteraction):
            raise TypeError(
                "SideEffectIntegrityEvaluator requires a "
                "SystemInteraction through interaction=."
            )

        expected_changes = kwargs.get(
            "expected_changes"
        )

        assessment = self.evaluate_interaction(
            interaction,
            expected_changes=expected_changes,
        )

        return Evaluation(
            result=assessment.result,
            rationale=assessment.rationale,
            confidence=assessment.confidence,
            property_name=property_name,
            property_expectation=property_expectation,
            evaluator=self.evaluator_name,
            evidence={
                "side_effect_assessment": {
                    "result": assessment.result.value,
                    "rationale": assessment.rationale,
                    "confidence": assessment.confidence,
                    "observations": list(
                        assessment.observations
                    ),
                    "evidence": assessment.evidence,
                },
                "comparison_changed": comparison.changed,
                "comparison_observations": list(
                    comparison.observations
                ),
            },
        )

    def evaluate_interaction(
        self,
        interaction: SystemInteraction,
        expected_changes: list[dict[str, Any]] | None = None,
    ) -> SideEffectAssessment:
        if expected_changes is None:
            return SideEffectAssessment(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "No expected side effects were supplied, "
                    "so side-effect integrity could not be evaluated."
                ),
                confidence="Low",
                evidence={
                    "expected_changes": [],
                    "observed_changes": interaction.metadata.get(
                        "state_changes",
                        [],
                    ),
                },
            )

        observed_changes = interaction.metadata.get(
            "state_changes"
        )

        if observed_changes is None:
            return SideEffectAssessment(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "No observable state changes were supplied "
                    "by the target."
                ),
                confidence="Low",
                evidence={
                    "expected_changes": list(
                        expected_changes
                    ),
                    "observed_changes": [],
                },
            )

        expected = list(expected_changes)
        observed = list(observed_changes)

        violations: list[dict[str, Any]] = []

        if observed != expected:
            violations.append(
                {
                    "type": "unexpected_state_change",
                    "expected": expected,
                    "observed": observed,
                }
            )

        if violations:
            return SideEffectAssessment(
                result=ExperimentResult.FAIL,
                rationale=(
                    "Observed system-state changes did not match "
                    "the side effects expected by the experiment."
                ),
                confidence="High",
                observations=[
                    "Observed state differs from the defined "
                    "expected state transition."
                ],
                evidence={
                    "expected_changes": expected,
                    "observed_changes": observed,
                    "violations": violations,
                },
            )

        return SideEffectAssessment(
            result=ExperimentResult.PASS,
            rationale=(
                "Observed system-state changes matched the "
                "expected side effects."
            ),
            confidence="High",
            observations=[
                "Observed state transition matched expectations."
            ],
            evidence={
                "expected_changes": expected,
                "observed_changes": observed,
                "violations": [],
            },
        )
