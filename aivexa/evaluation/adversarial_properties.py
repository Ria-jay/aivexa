from __future__ import annotations

from typing import Any, Callable

from aivexa.comparison.behavior import compare_responses
from aivexa.evaluation.adversarial import AdversarialCase
from aivexa.evaluation.agent_assessment import (
    AgentAssessmentEngine,
    AssessmentCheck,
)
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evaluation.security_checks import build_default_security_checks
from aivexa.evidence.system import SystemInteraction


class AdversarialPropertyRegistry:
    def __init__(self):
        self._checks: dict[str, AssessmentCheck] = {
            check.name: check
            for check in build_default_security_checks()
        }

        self._property_to_check = {
            "authorization_boundary_preservation": "authorization",
            "data_boundary_preservation": "data_boundary",
            "action_boundary_preservation": "action_boundary",
            "retrieval_integrity": "retrieval_integrity",
            "context_boundary_preservation": "context_boundary",
            "tool_invocation_integrity": "tool_invocation",
            "tool_result_boundary_preservation": "tool_result_boundary",
            "side_effect_integrity": "side_effect",
            "identity_boundary_preservation": "identity_boundary",
            "output_handling_integrity": "output_handling",
            "safety_boundary_preservation": None,
        }

    def get(
        self,
        property_name: str,
    ) -> Callable[
        [AdversarialCase, list[Any]],
        Evaluation,
    ] | None:
        check_name = self._property_to_check.get(property_name)

        if check_name is None:
            return None

        check = self._checks.get(check_name)

        if check is None:
            return None

        def evaluate_adversarial(
            case: AdversarialCase,
            observations: list[Any],
        ) -> Evaluation:
            if not observations:
                return Evaluation(
                    result=ExperimentResult.INCONCLUSIVE,
                    rationale=(
                        "No observations were available for "
                        "the adversarial property evaluation."
                    ),
                    confidence="Low",
                    property_name=case.property_name,
                    property_expectation=case.property_expectation,
                    evaluator="aivexa-adversarial-property-registry",
                    evidence={"observation_count": 0},
                )

            evaluations: list[Evaluation] = []

            for observation in observations:
                comparison = compare_responses(
                    observation.baseline_output,
                    observation.variant_output,
                )

                interaction = SystemInteraction(
                    experiment_id=(
                        f"adversarial-{observation.variant_id}"
                    ),
                    user_input=observation.variant_output,
                    system_output=observation.variant_output,
                    context=dict(observation.context),
                    observations=list(observation.observations),
                )

                evaluation = check.evaluator.evaluate(
                    comparison,
                    property_name=case.property_name,
                    property_expectation=(
                        case.property_expectation
                    ),
                    interaction=interaction,
                )

                evaluations.append(evaluation)

            if any(
                evaluation.result == ExperimentResult.FAIL
                for evaluation in evaluations
            ):
                result = ExperimentResult.FAIL
            elif any(
                evaluation.result == ExperimentResult.ANOMALY
                for evaluation in evaluations
            ):
                result = ExperimentResult.ANOMALY
            elif all(
                evaluation.result == ExperimentResult.PASS
                for evaluation in evaluations
            ):
                result = ExperimentResult.PASS
            elif any(
                evaluation.result == ExperimentResult.INCONCLUSIVE
                for evaluation in evaluations
            ):
                result = ExperimentResult.INCONCLUSIVE
            else:
                result = ExperimentResult.INCONCLUSIVE

            confidence_rank = {
                "Low": 1,
                "Medium": 2,
                "High": 3,
            }

            confidence = min(
                (
                    evaluation.confidence
                    for evaluation in evaluations
                ),
                key=lambda value: confidence_rank.get(
                    value,
                    1,
                ),
            )

            return Evaluation(
                result=result,
                rationale=(
                    "The adversarial observation was evaluated "
                    "using the registered AIVEXA security "
                    "property evaluator."
                ),
                confidence=confidence,
                property_name=case.property_name,
                property_expectation=case.property_expectation,
                evaluator="aivexa-adversarial-property-registry",
                evidence={
                    "individual_evaluations": [
                        {
                            "result": evaluation.result.value,
                            "confidence": evaluation.confidence,
                            "rationale": evaluation.rationale,
                            "property_name": evaluation.property_name,
                            "evaluator": evaluation.evaluator,
                            "evidence": evaluation.evidence,
                        }
                        for evaluation in evaluations
                    ],
                    "property_adapter": check_name,
                },
            )

        return evaluate_adversarial

    def get_check(
        self,
        property_name: str,
    ) -> AssessmentCheck | None:
        check_name = self._property_to_check.get(property_name)

        if check_name is None:
            return None

        return self._checks.get(check_name)

    def build_checks(
        self,
        property_name: str,
    ) -> list[AssessmentCheck]:
        check = self.get_check(property_name)

        if check is None:
            return []

        return [check]

    def evaluate_interaction(
        self,
        case: AdversarialCase,
        interaction: SystemInteraction,
        baseline_output: str | None = None,
    ) -> Evaluation:
        checks = self.build_checks(case.property_name)

        if not checks:
            return Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "The adversarial property does not have "
                    "a system-interaction evaluator registered "
                    "for this target type."
                ),
                confidence="Low",
                property_name=case.property_name,
                property_expectation=case.property_expectation,
                evaluator="aivexa-adversarial-property-registry",
                evidence={
                    "target_type": "system_interaction",
                    "evaluator_available": False,
                },
            )

        assessment = AgentAssessmentEngine().assess(
            assessment_id=(
                f"ADV-ASSESS-{interaction.experiment_id}"
            ),
            interaction=interaction,
            checks=checks,
            baseline_output=baseline_output,
        )

        return Evaluation(
            result=assessment.result,
            rationale=assessment.rationale,
            confidence=assessment.confidence,
            property_name=case.property_name,
            property_expectation=case.property_expectation,
            evaluator=AgentAssessmentEngine.engine_name,
            evidence={
                **assessment.evidence,
                "adversarial_case": case.case_id,
                "property_adapter": self._property_to_check[
                    case.property_name
                ],
            },
        )


def build_default_property_registry() -> AdversarialPropertyRegistry:
    return AdversarialPropertyRegistry()
