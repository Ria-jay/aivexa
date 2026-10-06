from dataclasses import dataclass, field
from typing import Any

from aivexa.evaluation.agent_assessment import AgentAssessment
from aivexa.evaluation.models import (
    Evaluation,
    ExperimentResult,
)


@dataclass(frozen=True)
class Finding:
    finding_id: str
    assessment_id: str
    experiment_id: str
    title: str
    result: ExperimentResult
    confidence: str
    affected_properties: list[str]
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)


class FindingBuilder:
    """
    Converts failed or anomalous AIVEXA evaluations into
    evidence-backed findings.

    PASS and INCONCLUSIVE results do not become findings.
    """

    def build(
        self,
        finding_id: str,
        assessment: AgentAssessment,
    ) -> Finding | None:
        if assessment.result not in {
            ExperimentResult.FAIL,
            ExperimentResult.ANOMALY,
        }:
            return None

        affected_properties = [
            evaluation.property_name
            for evaluation in assessment.evaluations
            if evaluation.result
            in {
                ExperimentResult.FAIL,
                ExperimentResult.ANOMALY,
            }
        ]

        if assessment.result == ExperimentResult.FAIL:
            title = "Agent security boundary violation"
        else:
            title = "Agent security boundary anomaly"

        return Finding(
            finding_id=finding_id,
            assessment_id=assessment.assessment_id,
            experiment_id=assessment.experiment_id,
            title=title,
            result=assessment.result,
            confidence=assessment.confidence,
            affected_properties=affected_properties,
            rationale=assessment.rationale,
            evidence={
                "assessment": assessment.evidence,
                "individual_evaluations": [
                    {
                        "result": evaluation.result.value,
                        "confidence": evaluation.confidence,
                        "property_name": evaluation.property_name,
                        "evaluator": evaluation.evaluator,
                        "rationale": evaluation.rationale,
                        "evidence": evaluation.evidence,
                    }
                    for evaluation in assessment.evaluations
                    if evaluation.result
                    in {
                        ExperimentResult.FAIL,
                        ExperimentResult.ANOMALY,
                    }
                ],
            },
        )

    def build_from_evaluation(
        self,
        finding_id: str,
        assessment_id: str,
        experiment_id: str,
        evaluation: Evaluation,
        evidence: dict[str, Any] | None = None,
    ) -> Finding | None:
        """
        Build a Finding directly from a generic AIVEXA Evaluation.

        This is used by model-safety and other non-Agent evaluation
        workflows while preserving the existing AgentAssessment API.
        """
        if evaluation.result not in {
            ExperimentResult.FAIL,
            ExperimentResult.ANOMALY,
        }:
            return None

        if evaluation.result == ExperimentResult.FAIL:
            title = "Model safety boundary violation"
        else:
            title = "Model safety boundary anomaly"

        merged_evidence = {
            "evaluation": {
                "result": evaluation.result.value,
                "confidence": evaluation.confidence,
                "property_name": evaluation.property_name,
                "property_expectation": (
                    evaluation.property_expectation
                ),
                "evaluator": evaluation.evaluator,
                "rationale": evaluation.rationale,
                "evidence": evaluation.evidence,
            }
        }

        if evidence:
            merged_evidence["execution"] = evidence

        return Finding(
            finding_id=finding_id,
            assessment_id=assessment_id,
            experiment_id=experiment_id,
            title=title,
            result=evaluation.result,
            confidence=evaluation.confidence,
            affected_properties=[
                evaluation.property_name
            ],
            rationale=evaluation.rationale,
            evidence=merged_evidence,
        )
