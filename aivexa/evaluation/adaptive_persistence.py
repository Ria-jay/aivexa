import json
from typing import Any

from aivexa.evaluation.adaptive_execution import AdaptiveExecution
from aivexa.storage.database import Database


class AdaptiveExperimentPersistence:
    """
    Persists an adaptive experiment using the existing AIVEXA database.

    The adaptive experiment keeps explicit lineage to the experiment
    that produced the probe.
    """

    def __init__(self, database: Database):
        self.database = database

    def save(
        self,
        execution: AdaptiveExecution,
    ) -> None:
        experiment = execution.experiment
        evaluation = execution.evaluation

        context = dict(experiment.context)

        context["adaptive_lineage"] = {
            "source_experiment_id": context.get("source_experiment_id"),
            "adaptive_experiment_id": experiment.experiment_id,
            "probe_id": execution.probe.probe_id,
        }

        context["adaptive_execution"] = {
            "baseline_output": execution.baseline_output,
            "adaptive_output": execution.adaptive_output,
            "comparison": execution.comparison,
            "evaluation": {
                "result": evaluation.result.value,
                "rationale": evaluation.rationale,
                "confidence": evaluation.confidence,
                "property_name": evaluation.property_name,
                "property_expectation": evaluation.property_expectation,
                "evaluator": evaluation.evaluator,
                "evidence": evaluation.evidence,
            },
        }

        self.database.save_experiment(
            {
                "experiment_id": experiment.experiment_id,
                "target": experiment.target,
                "objective": experiment.objective,
                "hypothesis": experiment.hypothesis,
                "intervention": experiment.intervention,
                "context": context,
                "created_at": experiment.created_at,
                "input_data": json.dumps(
                    {
                        "baseline": context.get("baseline_prompt"),
                        "adaptive": context.get("adaptive_prompt"),
                    }
                ),
                "output_data": execution.adaptive_output,
                "result": evaluation.result.value,
                "rationale": evaluation.rationale,
                "confidence": evaluation.confidence,
                "observations": execution.comparison["observations"],
                "property_name": evaluation.property_name,
                "property_expectation": evaluation.property_expectation,
                "comparison_group": context.get(
                    "source_experiment_id"
                ),
                "evaluator": evaluation.evaluator,
            }
        )
