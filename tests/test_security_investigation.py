from aivexa.evaluation.agent_assessment import AgentAssessment
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evaluation.security_follow_up import SecurityFollowUpPlanner
from aivexa.evaluation.security_investigation import (
    SecurityInvestigationBuilder,
)


def make_assessment() -> AgentAssessment:
    evaluation = Evaluation(
        result=ExperimentResult.FAIL,
        rationale="Unauthorized tool invocation observed.",
        confidence="High",
        property_name="tool_invocation_integrity",
        property_expectation="Only authorized tools may be invoked.",
        evaluator="aivexa-tool-invocation",
        evidence={
            "unauthorized_tool": "admin_delete",
        },
    )

    return AgentAssessment(
        assessment_id="AVX-P4-ASSESS-INV-001",
        experiment_id="AVX-P4-EXP-INV-001",
        result=ExperimentResult.FAIL,
        confidence="High",
        evaluations=[evaluation],
        rationale="Tool invocation boundary failed.",
        evidence={
            "failed_property": "tool_invocation_integrity",
        },
    )


def test_security_investigation_creates_follow_up_experiment():
    assessment = make_assessment()

    follow_up = SecurityFollowUpPlanner().plan(assessment)

    investigation = SecurityInvestigationBuilder().build(
        assessment=assessment,
        follow_up=follow_up,
        target="authorized-ai-application",
    )

    experiment = investigation.experiment

    assert (
        experiment.experiment_id
        == "AVX-P4-EXP-INV-001-FOLLOWUP-TOOL_INVOCATION_PROBE"
    )

    assert experiment.target == "authorized-ai-application"
    assert experiment.intervention == "TOOL_INVOCATION_PROBE"
    assert experiment.context["phase"] == 4
    assert experiment.context["security_investigation"] is True
    assert (
        experiment.context["source_experiment_id"]
        == "AVX-P4-EXP-INV-001"
    )
    assert (
        experiment.context["assessment_id"]
        == "AVX-P4-ASSESS-INV-001"
    )
    assert (
        "tool_invocation_integrity"
        in experiment.context["affected_properties"]
    )


def test_security_investigation_preserves_research_reasoning():
    assessment = make_assessment()

    follow_up = SecurityFollowUpPlanner().plan(assessment)

    investigation = SecurityInvestigationBuilder().build(
        assessment=assessment,
        follow_up=follow_up,
        target="authorized-ai-application",
    )

    experiment = investigation.experiment

    assert experiment.objective == follow_up.objective
    assert experiment.hypothesis == follow_up.hypothesis
    assert experiment.context["reason"] == follow_up.reason
    assert (
        experiment.context["suggested_experiments"]
        == list(follow_up.suggested_experiments)
    )
