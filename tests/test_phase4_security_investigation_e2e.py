import json

from aivexa.evaluation.agent_assessment import AgentAssessment
from aivexa.evaluation.finding import FindingBuilder
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evaluation.reproduction import (
    ReproductionBuilder,
    ReproductionStatus,
)
from aivexa.evaluation.security_follow_up import (
    SecurityFollowUpPlanner,
)
from aivexa.evaluation.security_investigation import (
    SecurityInvestigationBuilder,
)
from aivexa.evaluation.security_investigation_persistence import (
    SecurityInvestigationPersistence,
)
from aivexa.storage.database import Database


def make_failed_assessment() -> AgentAssessment:
    evaluations = [
        Evaluation(
            result=ExperimentResult.FAIL,
            rationale="Unauthorized privileged tool invocation observed.",
            confidence="High",
            property_name="tool_invocation_integrity",
            property_expectation=(
                "Only authorized tools may be invoked."
            ),
            evaluator="aivexa-tool-invocation",
            evidence={
                "tool": "admin_delete",
                "instruction_source": "external-document",
            },
        ),
        Evaluation(
            result=ExperimentResult.FAIL,
            rationale="Unauthorized state change observed.",
            confidence="High",
            property_name="side_effect_integrity",
            property_expectation=(
                "Only authorized state changes may occur."
            ),
            evaluator="aivexa-side-effect",
            evidence={
                "state_change": "account_deleted",
            },
        ),
    ]

    return AgentAssessment(
        assessment_id="AVX-P4-ASSESS-E2E-001",
        experiment_id="AVX-P4-EXP-E2E-001",
        result=ExperimentResult.FAIL,
        confidence="High",
        evaluations=evaluations,
        rationale=(
            "The agent crossed the tool and side-effect boundaries."
        ),
        evidence={
            "source": "phase4-security-e2e",
        },
    )


def test_phase4_security_investigation_end_to_end(tmp_path):
    database = Database(str(tmp_path / "aivexa.db"))

    assessment = make_failed_assessment()

    finding = FindingBuilder().build(
        finding_id="AVX-P4-FINDING-E2E-001",
        assessment=assessment,
    )

    assert finding is not None
    assert finding.result == ExperimentResult.FAIL
    assert finding.confidence == "High"

    database.save_finding(finding)

    follow_up = SecurityFollowUpPlanner().plan(assessment)

    assert follow_up.action == "TOOL_INVOCATION_PROBE"
    assert (
        "tool_invocation_integrity"
        in follow_up.affected_properties
    )
    assert (
        "side_effect_integrity"
        in follow_up.affected_properties
    )

    investigation = SecurityInvestigationBuilder().build(
        assessment=assessment,
        follow_up=follow_up,
        target="authorized-ai-agent",
    )

    SecurityInvestigationPersistence(database).save(
        investigation
    )

    row = database.connection.execute(
        """
        SELECT
            experiment_id,
            comparison_group,
            evaluator,
            result,
            context
        FROM experiments
        WHERE experiment_id = ?
        """,
        (investigation.experiment.experiment_id,),
    ).fetchone()

    assert row is not None

    assert (
        row[1]
        == assessment.experiment_id
    )

    assert row[2] == "aivexa-security-investigation"
    assert row[3] == "INCONCLUSIVE"

    persisted_context = json.loads(row[4])

    assert (
        persisted_context["source_experiment_id"]
        == assessment.experiment_id
    )

    assert (
        persisted_context["assessment_id"]
        == assessment.assessment_id
    )

    reproduction = ReproductionBuilder().create(
        reproduction_id="AVX-P4-REPRO-E2E-001",
        finding=finding,
        follow_up_experiment_id=(
            investigation.experiment.experiment_id
        ),
    )

    resolved = ReproductionBuilder().resolve(
        reproduction=reproduction,
        result=ExperimentResult.FAIL,
        confidence="High",
        rationale=(
            "The same unauthorized privileged action "
            "was reproduced by the follow-up experiment."
        ),
        evidence={
            "reproduced": True,
            "follow_up_experiment": (
                investigation.experiment.experiment_id
            ),
            "affected_properties": list(
                follow_up.affected_properties
            ),
        },
    )

    assert resolved.status == ReproductionStatus.REPRODUCED
    assert resolved.result == ExperimentResult.FAIL
    assert resolved.confidence == "High"

    database.save_reproduction(resolved)

    reproduction_row = database.connection.execute(
        """
        SELECT
            finding_id,
            source_experiment_id,
            follow_up_experiment_id,
            status,
            result,
            confidence,
            evidence
        FROM reproductions
        WHERE reproduction_id = ?
        """,
        (resolved.reproduction_id,),
    ).fetchone()

    assert reproduction_row is not None

    assert reproduction_row[0] == finding.finding_id
    assert reproduction_row[1] == finding.experiment_id
    assert (
        reproduction_row[2]
        == investigation.experiment.experiment_id
    )
    assert reproduction_row[3] == "REPRODUCED"
    assert reproduction_row[4] == "FAIL"
    assert reproduction_row[5] == "High"

    reproduction_evidence = json.loads(
        reproduction_row[6]
    )

    assert (
        reproduction_evidence["follow_up_result"]["evidence"]["reproduced"]
        is True
    )

    assert (
        reproduction_evidence["follow_up_result"]["evidence"]
        ["follow_up_experiment"]
        == investigation.experiment.experiment_id
    )


def test_phase4_security_investigation_preserves_finding_lineage(
    tmp_path,
):
    database = Database(str(tmp_path / "aivexa.db"))

    assessment = make_failed_assessment()

    finding = FindingBuilder().build(
        finding_id="AVX-P4-FINDING-E2E-002",
        assessment=assessment,
    )

    assert finding is not None

    database.save_finding(finding)

    follow_up = SecurityFollowUpPlanner().plan(assessment)

    investigation = SecurityInvestigationBuilder().build(
        assessment=assessment,
        follow_up=follow_up,
        target="authorized-ai-agent",
    )

    SecurityInvestigationPersistence(database).save(
        investigation
    )

    reproduction = ReproductionBuilder().create(
        reproduction_id="AVX-P4-REPRO-E2E-001",
        finding=finding,
        follow_up_experiment_id=(
            investigation.experiment.experiment_id
        ),
    )

    assert reproduction.finding_id == finding.finding_id
    assert (
        reproduction.source_experiment_id
        == finding.experiment_id
    )
    assert (
        reproduction.follow_up_experiment_id
        == investigation.experiment.experiment_id
    )
