from dataclasses import dataclass
from typing import Any

from aivexa.evaluation.agent_assessment import AgentAssessment
from aivexa.evaluation.security_follow_up import SecurityFollowUp
from aivexa.experiments.models import Experiment


@dataclass(frozen=True)
class SecurityInvestigation:
    source_experiment_id: str
    follow_up: SecurityFollowUp
    experiment: Experiment

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_experiment_id": self.source_experiment_id,
            "follow_up": self.follow_up.to_dict(),
            "experiment": {
                "experiment_id": self.experiment.experiment_id,
                "target": self.experiment.target,
                "objective": self.experiment.objective,
                "hypothesis": self.experiment.hypothesis,
                "intervention": self.experiment.intervention,
                "context": self.experiment.context,
                "created_at": self.experiment.created_at,
            },
        }


class SecurityInvestigationBuilder:
    """
    Converts a Phase 4 security assessment and its follow-up decision
    into a controlled AIVEXA experiment.

    This class creates the next experiment definition only.
    It does not execute the experiment.
    """

    def build(
        self,
        assessment: AgentAssessment,
        follow_up: SecurityFollowUp,
        target: str,
    ) -> SecurityInvestigation:
        source_experiment_id = assessment.experiment_id

        experiment_id = (
            f"{source_experiment_id}"
            f"-FOLLOWUP-{follow_up.action}"
        )

        experiment = Experiment(
            experiment_id=experiment_id,
            target=target,
            objective=follow_up.objective,
            hypothesis=follow_up.hypothesis,
            intervention=follow_up.action,
            context={
                "phase": 4,
                "security_investigation": True,
                "source_experiment_id": source_experiment_id,
                "assessment_id": assessment.assessment_id,
                "follow_up_action": follow_up.action,
                "priority": follow_up.priority,
                "affected_properties": list(
                    follow_up.affected_properties
                ),
                "reason": follow_up.reason,
                "suggested_experiments": list(
                    follow_up.suggested_experiments
                ),
                "source_assessment": {
                    "result": assessment.result.value,
                    "confidence": assessment.confidence,
                    "rationale": assessment.rationale,
                },
            },
        )

        return SecurityInvestigation(
            source_experiment_id=source_experiment_id,
            follow_up=follow_up,
            experiment=experiment,
        )
