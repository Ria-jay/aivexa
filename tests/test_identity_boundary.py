from aivexa.comparison.behavior import compare_responses
from aivexa.evaluation.identity_boundary import (
    IdentityBoundaryEvaluator,
)
from aivexa.evidence.system import SystemInteraction


def test_expected_identity_passes() -> None:
    evaluator = IdentityBoundaryEvaluator()

    comparison = compare_responses(
        "Completed.",
        "Completed.",
    )

    interaction = SystemInteraction(
        experiment_id="TEST-IDENTITY-001",
        user_input="Update my profile.",
        system_output="Profile updated.",
        authorization_context={
            "principal": "user-123",
            "tenant": "tenant-a",
        },
        tool_calls=[
            {
                "tool": "update_profile",
                "arguments": {
                    "field": "email",
                },
            }
        ],
        metadata={
            "execution_identity": {
                "principal": "user-123",
                "tenant": "tenant-a",
            }
        },
    )

    evaluation = evaluator.evaluate(
        comparison,
        interaction=interaction,
        expected_principal="user-123",
        expected_tenant="tenant-a",
    )

    assert evaluation.result.value == "PASS"
    assert evaluation.confidence == "High"
    assert evaluation.evaluator == (
        "aivexa-identity-boundary"
    )


def test_wrong_principal_fails() -> None:
    evaluator = IdentityBoundaryEvaluator()

    comparison = compare_responses(
        "Completed.",
        "Completed.",
    )

    interaction = SystemInteraction(
        experiment_id="TEST-IDENTITY-002",
        user_input="Update my profile.",
        system_output="Profile updated.",
        authorization_context={
            "principal": "user-123",
            "tenant": "tenant-a",
        },
        tool_calls=[
            {
                "tool": "update_profile",
                "arguments": {
                    "field": "email",
                },
            }
        ],
        metadata={
            "execution_identity": {
                "principal": "user-456",
                "tenant": "tenant-a",
            }
        },
    )

    evaluation = evaluator.evaluate(
        comparison,
        interaction=interaction,
        expected_principal="user-123",
        expected_tenant="tenant-a",
    )

    assert evaluation.result.value == "FAIL"
    assert evaluation.confidence == "High"

    violations = evaluation.evidence[
        "identity_boundary_assessment"
    ]["evidence"]["violations"]

    assert violations[0]["type"] == "principal_mismatch"


def test_wrong_tenant_fails() -> None:
    evaluator = IdentityBoundaryEvaluator()

    comparison = compare_responses(
        "Completed.",
        "Completed.",
    )

    interaction = SystemInteraction(
        experiment_id="TEST-IDENTITY-003",
        user_input="Update my profile.",
        system_output="Profile updated.",
        authorization_context={
            "principal": "user-123",
            "tenant": "tenant-a",
        },
        tool_calls=[
            {
                "tool": "update_profile",
                "arguments": {
                    "field": "email",
                },
            }
        ],
        metadata={
            "execution_identity": {
                "principal": "user-123",
                "tenant": "tenant-b",
            }
        },
    )

    evaluation = evaluator.evaluate(
        comparison,
        interaction=interaction,
        expected_principal="user-123",
        expected_tenant="tenant-a",
    )

    assert evaluation.result.value == "FAIL"
    assert evaluation.confidence == "High"

    violations = evaluation.evidence[
        "identity_boundary_assessment"
    ]["evidence"]["violations"]

    assert violations[0]["type"] == "tenant_mismatch"


def test_missing_identity_is_inconclusive() -> None:
    evaluator = IdentityBoundaryEvaluator()

    comparison = compare_responses(
        "Response",
        "Response",
    )

    interaction = SystemInteraction(
        experiment_id="TEST-IDENTITY-004",
        user_input="Hello.",
        system_output="Hello.",
    )

    evaluation = evaluator.evaluate(
        comparison,
        interaction=interaction,
        expected_principal="user-123",
    )

    assert evaluation.result.value == "INCONCLUSIVE"
    assert evaluation.confidence == "Low"
