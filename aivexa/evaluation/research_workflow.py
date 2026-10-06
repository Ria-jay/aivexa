from dataclasses import dataclass
from typing import Any

from aivexa.evaluation.agent_assessment import (
    AgentAssessment,
    AgentAssessmentEngine,
    AssessmentCheck,
)
from aivexa.evaluation.finding import Finding, FindingBuilder
from aivexa.evaluation.models import ExperimentResult
from aivexa.evaluation.reproduction import (
    Reproduction,
    ReproductionBuilder,
)
from aivexa.evaluation.security_follow_up import (
    SecurityFollowUp,
    SecurityFollowUpPlanner,
)
from aivexa.evaluation.security_investigation import (
    SecurityInvestigation,
    SecurityInvestigationBuilder,
)
from aivexa.evaluation.security_investigation_persistence import (
    SecurityInvestigationPersistence,
)
from aivexa.evidence.models import Evidence
from aivexa.evidence.system import SystemInteraction
from aivexa.storage.database import Database
from aivexa.targets.application import AIApplicationTarget


@dataclass(frozen=True)
class ResearchResult:
    experiment_id: str
    interaction: SystemInteraction
    assessment: AgentAssessment
    finding: Finding | None
    follow_up: SecurityFollowUp
    investigation: SecurityInvestigation
    evidence: Evidence
    reproduction: Reproduction | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "assessment": {
                "assessment_id": self.assessment.assessment_id,
                "result": self.assessment.result.value,
                "confidence": self.assessment.confidence,
                "rationale": self.assessment.rationale,
            },
            "finding": (
                None
                if self.finding is None
                else {
                    "finding_id": self.finding.finding_id,
                    "title": self.finding.title,
                    "result": self.finding.result.value,
                    "confidence": self.finding.confidence,
                    "affected_properties": (
                        self.finding.affected_properties
                    ),
                }
            ),
            "follow_up": self.follow_up.to_dict(),
            "investigation": self.investigation.to_dict(),
            "reproduction": (
                None
                if self.reproduction is None
                else {
                    "reproduction_id": (
                        self.reproduction.reproduction_id
                    ),
                    "status": (
                        self.reproduction.status.value
                    ),
                    "result": self.reproduction.result.value,
                    "confidence": self.reproduction.confidence,
                }
            ),
            "evidence": self.evidence.metadata,
        }


class ResearchWorkflow:
    """
    Researcher-facing Phase 4 workflow.

    Coordinates target execution, multi-property assessment,
    finding creation, follow-up planning, investigation persistence,
    and optional reproduction.
    """

    def __init__(
        self,
        target: AIApplicationTarget,
        database: Database,
    ):
        self.target = target
        self.database = database

    def assess(
        self,
        experiment_id: str,
        user_input: str,
        checks: list[AssessmentCheck],
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
        baseline_output: str | None = None,
        finding_id: str | None = None,
    ) -> ResearchResult:
        interaction = self.target.interact(
            experiment_id=experiment_id,
            user_input=user_input,
            context=context,
            authorization_context=authorization_context,
        )

        assessment = AgentAssessmentEngine().assess(
            assessment_id=f"{experiment_id}-ASSESSMENT",
            interaction=interaction,
            checks=checks,
            baseline_output=baseline_output,
        )

        self._ensure_experiment_exists(
            experiment_id=experiment_id,
            interaction=interaction,
            assessment=assessment,
        )

        self.database.save_assessment(assessment)

        finding = None

        if finding_id:
            finding = FindingBuilder().build(
                finding_id=finding_id,
                assessment=assessment,
            )

            if finding:
                self.database.save_finding(finding)

        follow_up = SecurityFollowUpPlanner().plan(
            assessment
        )

        investigation = SecurityInvestigationBuilder().build(
            assessment=assessment,
            follow_up=follow_up,
            target=self.target.name,
        )

        SecurityInvestigationPersistence(
            self.database
        ).save(investigation)

        evidence = Evidence(
            experiment_id=experiment_id,
            input_data=user_input,
            output_data=interaction.system_output,
            observations=list(interaction.observations),
            metadata={
                "research_workflow": True,
                "target": self.target.name,
                "assessment": assessment.evidence,
                "interaction": {
                    "retrieved_context": (
                        interaction.retrieved_context
                    ),
                    "tool_calls": interaction.tool_calls,
                    "authorization_context": (
                        interaction.authorization_context
                    ),
                    "metadata": interaction.metadata,
                },
                "follow_up": follow_up.to_dict(),
                "investigation": investigation.to_dict(),
            },
        )

        reproduction = None

        if finding:
            reproduction = ReproductionBuilder().create(
                reproduction_id=(
                    f"{finding.finding_id}-REPRO-001"
                ),
                finding=finding,
                follow_up_experiment_id=(
                    investigation.experiment.experiment_id
                ),
            )

        return ResearchResult(
            experiment_id=experiment_id,
            interaction=interaction,
            assessment=assessment,
            finding=finding,
            follow_up=follow_up,
            investigation=investigation,
            evidence=evidence,
            reproduction=reproduction,
        )

    def _ensure_experiment_exists(
        self,
        experiment_id: str,
        interaction: SystemInteraction,
        assessment: AgentAssessment,
    ) -> None:
        self.database.save_experiment(
            {
                "experiment_id": experiment_id,
                "target": self.target.name,
                "objective": (
                    "Phase 4 AI application and agent security assessment."
                ),
                "hypothesis": (
                    "The target will preserve the declared "
                    "security properties under controlled evaluation."
                ),
                "intervention": "RESEARCHER_ASSESSMENT",
                "context": dict(interaction.context),
                "created_at": interaction.observed_at,
                "input_data": interaction.user_input,
                "output_data": interaction.system_output,
                "result": assessment.result.value,
                "rationale": assessment.rationale,
                "confidence": assessment.confidence,
                "observations": list(interaction.observations),
                "property_name": "system_security_assessment",
                "property_expectation": (
                    "Declared AI application and agent security "
                    "boundaries remain preserved."
                ),
                "comparison_group": None,
                "evaluator": "aivexa-agent-assessment",
            }
        )
