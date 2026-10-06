from dataclasses import dataclass, field
from typing import Any

from aivexa.comparison.behavior import compare_responses
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evidence.system import SystemInteraction


@dataclass(frozen=True)
class AssessmentCheck:
    """
    Configuration for one evaluator within a system assessment.
    """
    name: str
    evaluator: Any
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentAssessment:
    """
    Complete system-level assessment composed from multiple
    independent evaluator results.
    """
    assessment_id: str
    experiment_id: str
    result: ExperimentResult
    confidence: str
    evaluations: list[Evaluation]
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)


class AgentAssessmentEngine:
    """
    Orchestrates independent security-property evaluators against
    one observed AI application or agent interaction.

    Individual evaluator results are preserved. The aggregate result
    never replaces the underlying evidence.
    """

    engine_name = "aivexa-agent-assessment"

    def assess(
        self,
        assessment_id: str,
        interaction: SystemInteraction,
        checks: list[AssessmentCheck],
        baseline_output: str | None = None,
    ) -> AgentAssessment:
        comparison_baseline = (
            baseline_output
            if baseline_output is not None
            else interaction.system_output
        )

        comparison = compare_responses(
            comparison_baseline,
            interaction.system_output,
        )

        evaluations: list[Evaluation] = []

        for check in checks:
            evaluation = self._evaluate_check(
                check=check,
                comparison=comparison,
                interaction=interaction,
            )
            evaluations.append(evaluation)

        result = self._aggregate_result(evaluations)
        confidence = self._aggregate_confidence(evaluations)
        rationale = self._build_rationale(evaluations)

        evidence = {
            "engine": self.engine_name,
            "assessment_id": assessment_id,
            "experiment_id": interaction.experiment_id,
            "comparison": {
                "changed": comparison.changed,
                "observations": list(
                    comparison.observations
                ),
            },
            "interaction": {
                "experiment_id": interaction.experiment_id,
                "user_input": interaction.user_input,
                "system_output": interaction.system_output,
                "context": dict(interaction.context),
                "retrieved_context": list(
                    interaction.retrieved_context
                ),
                "tool_calls": list(
                    interaction.tool_calls
                ),
                "authorization_context": dict(
                    interaction.authorization_context
                ),
                "observations": list(
                    interaction.observations
                ),
                "metadata": dict(interaction.metadata),
                "observed_at": interaction.observed_at,
            },
            "evaluations": [
                {
                    "check": checks[index].name,
                    "result": evaluation.result.value,
                    "confidence": evaluation.confidence,
                    "property_name": evaluation.property_name,
                    "evaluator": evaluation.evaluator,
                    "rationale": evaluation.rationale,
                    "evidence": evaluation.evidence,
                }
                for index, evaluation in enumerate(evaluations)
            ],
        }

        return AgentAssessment(
            assessment_id=assessment_id,
            experiment_id=interaction.experiment_id,
            result=result,
            confidence=confidence,
            evaluations=evaluations,
            rationale=rationale,
            evidence=evidence,
        )

    @staticmethod
    def _evaluate_check(
        check: AssessmentCheck,
        comparison: Any,
        interaction: SystemInteraction,
    ) -> Evaluation:
        """
        Execute an evaluator through the AIVEXA evaluator contract.

        Application/security evaluators receive the behavioral comparison
        plus the observed interaction. Evaluators whose primary contract
        is interaction-only may omit the comparison parameter.
        """

        evaluator = check.evaluator
        kwargs = dict(check.kwargs)

        try:
            return evaluator.evaluate(
                comparison,
                interaction=interaction,
                **kwargs,
            )
        except TypeError as exc:
            message = str(exc)

            if "multiple values for argument 'interaction'" not in message:
                raise

            return evaluator.evaluate(
                interaction,
                **kwargs,
            )

    @staticmethod
    def _aggregate_result(
        evaluations: list[Evaluation],
    ) -> ExperimentResult:
        if not evaluations:
            return ExperimentResult.INCONCLUSIVE

        results = {
            evaluation.result
            for evaluation in evaluations
        }

        if ExperimentResult.FAIL in results:
            return ExperimentResult.FAIL

        if ExperimentResult.ANOMALY in results:
            return ExperimentResult.ANOMALY

        if results == {ExperimentResult.PASS}:
            return ExperimentResult.PASS

        if (
            ExperimentResult.PASS in results
            and ExperimentResult.INCONCLUSIVE in results
        ):
            return ExperimentResult.INCONCLUSIVE

        if ExperimentResult.INCONCLUSIVE in results:
            return ExperimentResult.INCONCLUSIVE

        if results == {ExperimentResult.NOT_APPLICABLE}:
            return ExperimentResult.NOT_APPLICABLE

        return ExperimentResult.INCONCLUSIVE

    @staticmethod
    def _aggregate_confidence(
        evaluations: list[Evaluation],
    ) -> str:
        if not evaluations:
            return "Low"

        confidence_levels = {
            "High": 3,
            "Medium": 2,
            "Low": 1,
        }

        values = [
            confidence_levels.get(
                evaluation.confidence,
                1,
            )
            for evaluation in evaluations
        ]

        minimum = min(values)

        if minimum == 3:
            return "High"

        if minimum == 2:
            return "Medium"

        return "Low"

    @staticmethod
    def _build_rationale(
        evaluations: list[Evaluation],
    ) -> str:
        if not evaluations:
            return (
                "No evaluator results were produced for "
                "the assessment."
            )

        failures = [
            evaluation.property_name
            for evaluation in evaluations
            if evaluation.result == ExperimentResult.FAIL
        ]

        anomalies = [
            evaluation.property_name
            for evaluation in evaluations
            if evaluation.result == ExperimentResult.ANOMALY
        ]

        inconclusive = [
            evaluation.property_name
            for evaluation in evaluations
            if evaluation.result == ExperimentResult.INCONCLUSIVE
        ]

        if failures:
            return (
                "One or more evaluated security properties "
                "failed: "
                + ", ".join(failures)
                + "."
            )

        if anomalies:
            return (
                "One or more evaluated security properties "
                "produced anomalies: "
                + ", ".join(anomalies)
                + "."
            )

        if inconclusive:
            return (
                "No evaluated property failed, but one or more "
                "properties were inconclusive: "
                + ", ".join(inconclusive)
                + "."
            )

        return (
            "All evaluated security properties satisfied the "
            "expectations defined for this assessment."
        )
