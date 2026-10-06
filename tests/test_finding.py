from aivexa.evaluation.agent_assessment import AgentAssessment
from aivexa.evaluation.finding import FindingBuilder
from aivexa.evaluation.models import Evaluation, ExperimentResult


def build_assessment(
    result: ExperimentResult,
) -> AgentAssessment:
    failed_evaluation = Evaluation(
        result=ExperimentResult.FAIL,
        rationale="Unauthorized action observed.",
        confidence="High",
        property_name="action_boundary_preservation",
        property_expectation="Actions remain within the allowed boundary.",
        evaluator="aivexa-action-boundary",
        evidence={
            "violations": [
                {
                    "type": "unauthorized_action",
                    "tool": "admin_delete",
                }
            ]
        },
    )

    passing_evaluation = Evaluation(
        result=ExperimentResult.PASS,
        rationale="Identity remained within the expected boundary.",
        confidence="High",
        property_name="identity_boundary_preservation",
        property_expectation="Execution identity remains authorized.",
        evaluator="aivexa-identity-boundary",
        evidence={},
    )

    return AgentAssessment(
        assessment_id="ASSESSMENT-FINDING-001",
        experiment_id="EXPERIMENT-FINDING-001",
        result=result,
        confidence="High",
        evaluations=[
            failed_evaluation,
            passing_evaluation,
        ],
        rationale=(
            "One or more evaluated security properties failed."
        ),
        evidence={
            "engine": "aivexa-agent-assessment",
        },
    )


def test_failed_assessment_creates_finding() -> None:
    finding = FindingBuilder().build(
        finding_id="FINDING-001",
        assessment=build_assessment(
            ExperimentResult.FAIL
        ),
    )

    assert finding is not None
    assert finding.finding_id == "FINDING-001"
    assert finding.assessment_id == "ASSESSMENT-FINDING-001"
    assert finding.experiment_id == "EXPERIMENT-FINDING-001"
    assert finding.result == ExperimentResult.FAIL
    assert finding.confidence == "High"

    assert finding.affected_properties == [
        "action_boundary_preservation"
    ]

    assert (
        finding.evidence["individual_evaluations"][0][
            "property_name"
        ]
        == "action_boundary_preservation"
    )


def test_anomalous_assessment_creates_finding() -> None:
    finding = FindingBuilder().build(
        finding_id="FINDING-002",
        assessment=build_assessment(
            ExperimentResult.ANOMALY
        ),
    )

    assert finding is not None
    assert finding.result == ExperimentResult.ANOMALY
    assert finding.title == "Agent security boundary anomaly"


def test_pass_does_not_create_finding() -> None:
    finding = FindingBuilder().build(
        finding_id="FINDING-003",
        assessment=build_assessment(
            ExperimentResult.PASS
        ),
    )

    assert finding is None


def test_inconclusive_does_not_create_finding() -> None:
    finding = FindingBuilder().build(
        finding_id="FINDING-004",
        assessment=build_assessment(
            ExperimentResult.INCONCLUSIVE
        ),
    )

    assert finding is None
