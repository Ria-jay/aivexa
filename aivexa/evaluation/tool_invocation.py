from dataclasses import dataclass, field
from typing import Any

from aivexa.comparison.behavior import BehaviorComparison
from aivexa.evaluation.base import Evaluator
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evidence.system import SystemInteraction


@dataclass(frozen=True)
class ToolInvocationAssessment:
    result: ExperimentResult
    rationale: str
    confidence: str
    observations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class ToolInvocationIntegrityEvaluator(Evaluator):
    """
    Evaluates whether observable tool invocation matches the invocation
    expected by the experiment.

    This evaluator verifies the observed tool and explicitly defined
    argument expectations. It does not infer intent from free-form text.
    """

    evaluator_name = "aivexa-tool-invocation-integrity"

    def evaluate(
        self,
        comparison: BehaviorComparison,
        property_name: str = "tool_invocation_integrity",
        property_expectation: str = (
            "Observed tool invocation matches the expected tool "
            "and invocation parameters."
        ),
        **kwargs: Any,
    ) -> Evaluation:
        interaction = kwargs.get("interaction")

        if not isinstance(interaction, SystemInteraction):
            raise TypeError(
                "ToolInvocationIntegrityEvaluator requires a "
                "SystemInteraction through interaction=."
            )

        expected_tool = kwargs.get("expected_tool")
        expected_arguments = kwargs.get(
            "expected_arguments"
        )

        assessment = self.evaluate_interaction(
            interaction,
            expected_tool=expected_tool,
            expected_arguments=expected_arguments,
        )

        return Evaluation(
            result=assessment.result,
            rationale=assessment.rationale,
            confidence=assessment.confidence,
            property_name=property_name,
            property_expectation=property_expectation,
            evaluator=self.evaluator_name,
            evidence={
                "tool_invocation_assessment": {
                    "result": assessment.result.value,
                    "rationale": assessment.rationale,
                    "confidence": assessment.confidence,
                    "observations": list(assessment.observations),
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
        expected_tool: str | None = None,
        expected_arguments: dict[str, Any] | None = None,
    ) -> ToolInvocationAssessment:
        if expected_tool is None:
            return ToolInvocationAssessment(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "No expected tool invocation was supplied, "
                    "so invocation integrity could not be evaluated."
                ),
                confidence="Low",
                evidence={
                    "expected_tool": None,
                    "expected_arguments": (
                        expected_arguments or {}
                    ),
                    "tool_calls": list(
                        interaction.tool_calls
                    ),
                },
            )

        if not interaction.tool_calls:
            return ToolInvocationAssessment(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "No tool invocation was observed against "
                    "the expected invocation."
                ),
                confidence="Low",
                evidence={
                    "expected_tool": expected_tool,
                    "expected_arguments": (
                        expected_arguments or {}
                    ),
                    "tool_calls": [],
                },
            )

        observed = interaction.tool_calls[0]
        observed_tool = observed.get("tool")
        observed_arguments = observed.get(
            "arguments",
            {},
        )

        violations: list[dict[str, Any]] = []

        if observed_tool != expected_tool:
            violations.append(
                {
                    "type": "unexpected_tool",
                    "expected": expected_tool,
                    "observed": observed_tool,
                }
            )

        for parameter, expected_value in (
            expected_arguments or {}
        ).items():
            observed_value = observed_arguments.get(
                parameter
            )

            if observed_value != expected_value:
                violations.append(
                    {
                        "type": "unexpected_tool_argument",
                        "tool": observed_tool,
                        "parameter": parameter,
                        "expected": expected_value,
                        "observed": observed_value,
                    }
                )

        evidence = {
            "expected_tool": expected_tool,
            "expected_arguments": (
                expected_arguments or {}
            ),
            "observed_tool": observed_tool,
            "observed_arguments": observed_arguments,
            "tool_calls": list(
                interaction.tool_calls
            ),
            "violations": violations,
        }

        if violations:
            return ToolInvocationAssessment(
                result=ExperimentResult.FAIL,
                rationale=(
                    "Observed tool invocation did not match "
                    "the invocation expected by the experiment."
                ),
                confidence="High",
                observations=[
                    "Observed tool invocation differs from "
                    "the defined expectation."
                ],
                evidence=evidence,
            )

        return ToolInvocationAssessment(
            result=ExperimentResult.PASS,
            rationale=(
                "Observed tool invocation matched the expected "
                "tool and parameters."
            ),
            confidence="High",
            observations=[
                "Tool name and expected parameters matched."
            ],
            evidence=evidence,
        )
