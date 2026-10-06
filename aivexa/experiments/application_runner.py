from typing import Any

from aivexa.evaluation.base import Evaluator
from aivexa.evaluation.models import Evaluation
from aivexa.evidence.models import Evidence
from aivexa.evidence.system import SystemInteraction
from aivexa.targets.application import AIApplicationTarget
from aivexa.comparison.behavior import compare_responses


class ApplicationExperimentRunner:
    """
    Executes a controlled experiment against an AI application target
    and evaluates the resulting observable system behavior.

    The runner coordinates existing AIVEXA components. It does not
    implement security judgments itself.
    """

    def __init__(
        self,
        target: AIApplicationTarget,
        evaluator: Evaluator,
    ):
        self.target = target
        self.evaluator = evaluator

    def run(
        self,
        experiment_id: str,
        user_input: str,
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
        baseline_output: str | None = None,
    ) -> tuple[SystemInteraction, Evaluation, Evidence]:
        interaction = self.target.interact(
            experiment_id=experiment_id,
            user_input=user_input,
            context=context,
            authorization_context=authorization_context,
        )

        comparison_baseline = (
            baseline_output
            if baseline_output is not None
            else interaction.system_output
        )

        comparison = compare_responses(
            comparison_baseline,
            interaction.system_output,
        )

        evaluation = self.evaluator.evaluate(
            comparison,
            interaction=interaction,
        )

        evidence = Evidence(
            experiment_id=experiment_id,
            input_data=user_input,
            output_data=interaction.system_output,
            observations=list(interaction.observations),
            metadata={
                "target_type": "ai_application",
                "target_name": self.target.name,
                "endpoint": self.target.endpoint,
                "context": dict(interaction.context),
                "retrieved_context": list(
                    interaction.retrieved_context
                ),
                "tool_calls": list(interaction.tool_calls),
                "authorization_context": dict(
                    interaction.authorization_context
                ),
                "system_interaction": {
                    "experiment_id": interaction.experiment_id,
                    "user_input": interaction.user_input,
                    "system_output": interaction.system_output,
                    "context": dict(interaction.context),
                    "retrieved_context": list(
                        interaction.retrieved_context
                    ),
                    "tool_calls": list(interaction.tool_calls),
                    "authorization_context": dict(
                        interaction.authorization_context
                    ),
                    "observations": list(interaction.observations),
                    "metadata": dict(interaction.metadata),
                    "observed_at": interaction.observed_at,
                },
                "evaluation": {
                    "result": evaluation.result.value,
                    "rationale": evaluation.rationale,
                    "confidence": evaluation.confidence,
                    "property_name": evaluation.property_name,
                    "property_expectation": (
                        evaluation.property_expectation
                    ),
                    "evaluator": evaluation.evaluator,
                    "evidence": evaluation.evidence,
                },
            },
        )

        return interaction, evaluation, evidence
