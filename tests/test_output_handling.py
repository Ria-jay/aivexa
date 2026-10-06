from aivexa.comparison.behavior import compare_responses
from aivexa.evaluation.models import ExperimentResult
from aivexa.evaluation.output_handling import (
    OutputHandlingEvaluator,
)
from aivexa.evidence.system import SystemInteraction


def interaction(output: str, **metadata):
    return SystemInteraction(
        experiment_id="AVX-P4-OUTPUT-001",
        user_input="show information",
        system_output=output,
        metadata=metadata,
    )


def comparison(output: str):
    return compare_responses(output, output)


def test_safe_output_passes():
    evaluation = OutputHandlingEvaluator().evaluate(
        comparison("Public information only."),
        interaction=interaction("Public information only."),
        forbidden_values=["ADMIN-SECRET-123"],
        required_markers=["Public"],
    )

    assert evaluation.result == ExperimentResult.PASS
    assert evaluation.confidence == "High"


def test_forbidden_output_fails():
    evaluation = OutputHandlingEvaluator().evaluate(
        comparison("The secret is ADMIN-SECRET-123."),
        interaction=interaction(
            "The secret is ADMIN-SECRET-123."
        ),
        forbidden_values=["ADMIN-SECRET-123"],
    )

    assert evaluation.result == ExperimentResult.FAIL
    assert evaluation.evidence["violations"][0]["type"] == (
        "forbidden_output_value"
    )


def test_missing_required_marker_fails():
    evaluation = OutputHandlingEvaluator().evaluate(
        comparison("Unexpected response."),
        interaction=interaction("Unexpected response."),
        required_markers=["Approved"],
    )

    assert evaluation.result == ExperimentResult.FAIL


def test_explicit_application_violation_fails():
    evaluation = OutputHandlingEvaluator().evaluate(
        comparison("Response"),
        interaction=interaction(
            "Response",
            output_policy_violation={
                "reason": "sensitive_data_disclosure",
                "field": "customer_email",
            },
        ),
    )

    assert evaluation.result == ExperimentResult.FAIL
    assert evaluation.evidence["violations"][0]["type"] == (
        "explicit_output_policy_violation"
    )


def test_missing_policy_is_inconclusive():
    evaluation = OutputHandlingEvaluator().evaluate(
        comparison("Response"),
        interaction("Response"),
    )

    assert evaluation.result == ExperimentResult.INCONCLUSIVE
    assert evaluation.confidence == "Low"
