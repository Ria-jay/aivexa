from aivexa.evaluation.agent_assessment import (
    AgentAssessment,
)
from aivexa.evaluation.models import (
    Evaluation,
    ExperimentResult,
)
from aivexa.evaluation.security_follow_up import (
    SecurityFollowUpPlanner,
)


def make_evaluation(
    property_name: str,
    result: ExperimentResult,
) -> Evaluation:
    return Evaluation(
        result=result,
        rationale="test",
        confidence="High",
        property_name=property_name,
        property_expectation="boundary preserved",
        evaluator="test-evaluator",
        evidence={},
    )


def make_assessment(
    result: ExperimentResult,
    evaluations: list[Evaluation],
) -> AgentAssessment:
    return AgentAssessment(
        assessment_id="AVX-P4-ASSESS-001",
        experiment_id="AVX-P4-EXP-001",
        result=result,
        confidence="High",
        evaluations=evaluations,
        rationale="test assessment",
        evidence={},
    )


def test_failed_tool_invocation_creates_security_follow_up():
    assessment = make_assessment(
        ExperimentResult.FAIL,
        [
            make_evaluation(
                "tool_invocation_integrity",
                ExperimentResult.FAIL,
            )
        ],
    )

    follow_up = SecurityFollowUpPlanner().plan(assessment)

    assert follow_up.action == "TOOL_INVOCATION_PROBE"
    assert follow_up.priority == "HIGH"
    assert "tool_invocation_integrity" in follow_up.affected_properties
    assert len(follow_up.suggested_experiments) == 3


def test_multiple_failed_properties_are_preserved():
    assessment = make_assessment(
        ExperimentResult.FAIL,
        [
            make_evaluation(
                "action_boundary_preservation",
                ExperimentResult.FAIL,
            ),
            make_evaluation(
                "context_boundary_preservation",
                ExperimentResult.FAIL,
            ),
            make_evaluation(
                "side_effect_integrity",
                ExperimentResult.FAIL,
            ),
        ],
    )

    follow_up = SecurityFollowUpPlanner().plan(assessment)

    assert follow_up.action == "ACTION_PROBE"
    assert follow_up.affected_properties == (
        "action_boundary_preservation",
        "context_boundary_preservation",
        "side_effect_integrity",
    )


def test_inconclusive_assessment_requests_more_evidence():
    assessment = make_assessment(
        ExperimentResult.INCONCLUSIVE,
        [],
    )

    follow_up = SecurityFollowUpPlanner().plan(assessment)

    assert follow_up.action == "UNCERTAINTY_REDUCTION"
    assert follow_up.priority == "MEDIUM"


def test_secure_assessment_requests_regression_check():
    assessment = make_assessment(
        ExperimentResult.PASS,
        [
            make_evaluation(
                "authorization_boundary_preservation",
                ExperimentResult.PASS,
            ),
        ],
    )

    follow_up = SecurityFollowUpPlanner().plan(assessment)

    assert follow_up.action == "REGRESSION_CHECK"
    assert follow_up.priority == "LOW"
