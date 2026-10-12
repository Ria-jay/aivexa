from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aivexa.evaluation.agent_assessment import (
    AgentAssessment,
    AgentAssessmentEngine,
    AssessmentCheck,
)
from aivexa.evaluation.finding import (
    Finding,
    FindingBuilder,
)
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
from aivexa.evaluation.system_adaptive import (
    SystemAdaptiveController,
    SystemAdaptiveRun,
)
from aivexa.experiments.application_runner import (
    ApplicationExperimentRunner,
)
from aivexa.storage.database import Database
from aivexa.targets.application import AIApplicationTarget


@dataclass(frozen=True)
class ResearchExecution:
    assessment: AgentAssessment
    finding: Finding | None
    follow_up: SecurityFollowUp | None
    investigation: SecurityInvestigation | None


@dataclass(frozen=True)
class ReproductionExecution:
    reproduction: Reproduction
    assessment: AgentAssessment


class AuthorizedResearchRunner:
    """
    Researcher-facing Phase 4 execution workflow.

    Flow:

        authorized target
            -> interaction
            -> assessment
            -> finding
            -> investigation
            -> optional follow-up
            -> reproduction
    """

    def __init__(
        self,
        target: AIApplicationTarget,
        database: Database,
        max_adaptive_cycles: int = 0,
    ):
        if max_adaptive_cycles < 0:
            raise ValueError(
                "max_adaptive_cycles cannot be negative."
            )

        self.target = target
        self.database = database
        self.max_adaptive_cycles = max_adaptive_cycles
        self.assessment_engine = AgentAssessmentEngine()
        self.finding_builder = FindingBuilder()
        self.follow_up_planner = SecurityFollowUpPlanner()
        self.investigation_builder = SecurityInvestigationBuilder()
        self.investigation_persistence = (
            SecurityInvestigationPersistence(database)
        )

    def assess(
        self,
        experiment_id: str,
        user_input: str,
        checks: list[AssessmentCheck],
        authorization_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        baseline_output: str | None = None,
    ) -> ResearchExecution:
        interaction = self.target.interact(
            experiment_id=experiment_id,
            user_input=user_input,
            context=context,
            authorization_context=authorization_context,
        )

        assessment = self.assessment_engine.assess(
            assessment_id=f"{experiment_id}-ASSESSMENT",
            interaction=interaction,
            checks=checks,
            baseline_output=baseline_output,
        )

        evidence = assessment.evidence.get(
            "interaction",
            {},
        )

        self.database.save_experiment(
            {
                "experiment_id": experiment_id,
                "target": self.target.name,
                "objective": (
                    "Authorized AI application security assessment."
                ),
                "hypothesis": (
                    "The target will preserve its declared "
                    "security boundaries under controlled testing."
                ),
                "intervention": user_input,
                "context": {
                    "research_workflow": "authorized",
                    "target_type": "ai_application",
                    **dict(context or {}),
                },
                "created_at": interaction.observed_at,
                "input_data": interaction.user_input,
                "output_data": interaction.system_output,
                "result": assessment.result.value,
                "rationale": assessment.rationale,
                "confidence": assessment.confidence,
                "observations": interaction.observations,
                "property_name": (
                    assessment.evaluations[0].property_name
                    if assessment.evaluations
                    else "system_security_assessment"
                ),
                "property_expectation": (
                    "Declared AI application security boundaries "
                    "should remain preserved."
                ),
                "comparison_group": None,
                "evaluator": (
                    "aivexa-agent-assessment"
                ),
            }
        )

        self.database.save_assessment(assessment)

        finding = self.finding_builder.build(
            finding_id=f"{experiment_id}-FINDING",
            assessment=assessment,
        )

        if finding is None:
            return ResearchExecution(
                assessment=assessment,
                finding=None,
                follow_up=None,
                investigation=None,
            )

        self.database.save_finding(finding)

        follow_up = self.follow_up_planner.plan(
            assessment=assessment,
        )

        investigation = self.investigation_builder.build(
            assessment=assessment,
            follow_up=follow_up,
            target=self.target.name,
        )

        self.investigation_persistence.save(
            investigation
        )

        return ResearchExecution(
            assessment=assessment,
            finding=finding,
            follow_up=follow_up,
            investigation=investigation,
        )

    def investigate(
        self,
        source_experiment_id: str,
        source_input: str,
        checks: list[AssessmentCheck],
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
        baseline_output: str | None = None,
        max_cycles: int | None = None,
    ) -> SystemAdaptiveRun:
        """
        Execute a bounded, evidence-driven adaptive follow-up for a
        system target and persist each executed follow-up experiment.
        """
        if baseline_output is None:
            row = self.database.connection.execute(
                """
                SELECT output_data
                FROM experiments
                WHERE experiment_id = ?
                """,
                (source_experiment_id,),
            ).fetchone()

            if row is not None:
                baseline_output = row[0]

        source_finding = None

        finding_row = self.database.connection.execute(
            """
            SELECT
                finding_id,
                assessment_id,
                experiment_id,
                title,
                result,
                confidence,
                affected_properties,
                rationale,
                evidence
            FROM findings
            WHERE experiment_id = ?
            ORDER BY created_at
            LIMIT 1
            """,
            (source_experiment_id,),
        ).fetchone()

        if finding_row is not None:
            source_finding = Finding(
                finding_id=finding_row[0],
                assessment_id=finding_row[1],
                experiment_id=finding_row[2],
                title=finding_row[3],
                result=ExperimentResult(finding_row[4]),
                confidence=finding_row[5],
                affected_properties=json.loads(finding_row[6]),
                rationale=finding_row[7],
                evidence=json.loads(finding_row[8]),
            )

        controller = SystemAdaptiveController(
            target=self.target,
            database=self.database,
            checks=checks,
            max_cycles=(
                max_cycles
                if max_cycles is not None
                else self.max_adaptive_cycles
            ),
        )

        return controller.run(
            source_experiment_id=source_experiment_id,
            source_input=source_input,
            baseline_output=baseline_output,
            context=context,
            authorization_context=authorization_context,
            source_finding=source_finding,
        )

    def reproduce(
        self,
        finding_id: str,
        follow_up_experiment_id: str,
        user_input: str,
        checks: list[AssessmentCheck],
        authorization_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        baseline_output: str | None = None,
    ) -> ReproductionExecution:
        row = self.database.connection.execute(
            """
            SELECT
                finding_id,
                assessment_id,
                experiment_id,
                title,
                result,
                confidence,
                affected_properties,
                rationale,
                evidence
            FROM findings
            WHERE finding_id = ?
            """,
            (finding_id,),
        ).fetchone()

        if row is None:
            raise KeyError(
                f"Finding '{finding_id}' was not found."
            )

        finding = Finding(
            finding_id=row[0],
            assessment_id=row[1],
            experiment_id=row[2],
            title=row[3],
            result=ExperimentResult(row[4]),
            confidence=row[5],
            affected_properties=json.loads(row[6]),
            rationale=row[7],
            evidence=json.loads(row[8]),
        )

        interaction = self.target.interact(
            experiment_id=follow_up_experiment_id,
            user_input=user_input,
            context=context,
            authorization_context=authorization_context,
        )

        assessment = self.assessment_engine.assess(
            assessment_id=(
                f"{follow_up_experiment_id}-ASSESSMENT"
            ),
            interaction=interaction,
            checks=checks,
            baseline_output=baseline_output,
        )

        self.database.save_experiment(
            {
                "experiment_id": follow_up_experiment_id,
                "target": self.target.name,
                "objective": (
                    "Controlled follow-up reproduction of a "
                    "previously observed security finding."
                ),
                "hypothesis": (
                    "The previously observed security behavior "
                    "will recur under controlled follow-up testing."
                ),
                "intervention": user_input,
                "context": {
                    "research_workflow": "reproduction",
                    "source_finding_id": finding_id,
                    "source_experiment_id": finding.experiment_id,
                    **dict(context or {}),
                },
                "created_at": interaction.observed_at,
                "input_data": interaction.user_input,
                "output_data": interaction.system_output,
                "result": assessment.result.value,
                "rationale": assessment.rationale,
                "confidence": assessment.confidence,
                "observations": interaction.observations,
                "property_name": (
                    assessment.evaluations[0].property_name
                    if assessment.evaluations
                    else "reproduction"
                ),
                "property_expectation": (
                    "Previously observed behavior should be "
                    "reproducible under controlled variation."
                ),
                "comparison_group": finding.experiment_id,
                "evaluator": "aivexa-reproduction",
            }
        )

        self.database.save_assessment(assessment)

        reproduction_builder = ReproductionBuilder()

        reproduction = reproduction_builder.create(
            reproduction_id=(
                f"{finding_id}-REPRODUCTION"
            ),
            finding=finding,
            follow_up_experiment_id=(
                follow_up_experiment_id
            ),
        )

        reproduction = reproduction_builder.resolve(
            reproduction=reproduction,
            result=assessment.result,
            confidence=assessment.confidence,
            rationale=assessment.rationale,
            evidence={
                "reproduced": (
                    assessment.result
                    in {
                        ExperimentResult.FAIL,
                        ExperimentResult.ANOMALY,
                    }
                ),
                "assessment_id": assessment.assessment_id,
                "evaluations": [
                    {
                        "property": evaluation.property_name,
                        "result": evaluation.result.value,
                        "confidence": evaluation.confidence,
                        "rationale": evaluation.rationale,
                    }
                    for evaluation in assessment.evaluations
                ],
            },
        )

        self.database.save_reproduction(
            reproduction
        )

        return ReproductionExecution(
            reproduction=reproduction,
            assessment=assessment,
        )
