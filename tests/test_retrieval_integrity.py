from aivexa.comparison.behavior import compare_responses
from aivexa.evaluation.retrieval_integrity import (
    RetrievalIntegrityEvaluator,
)
from aivexa.evidence.system import SystemInteraction


def test_expected_retrieval_passes() -> None:
    evaluator = RetrievalIntegrityEvaluator()

    comparison = compare_responses(
        "Expected response",
        "Expected response",
    )

    interaction = SystemInteraction(
        experiment_id="TEST-RETRIEVAL-001",
        user_input="Find the public policy.",
        system_output="Here is the policy.",
        retrieved_context=[
            {
                "source": "public-policies",
                "scope": "public",
                "content": "Policy content",
            }
        ],
    )

    evaluation = evaluator.evaluate(
        comparison,
        interaction=interaction,
        expected_sources=["public-policies"],
        expected_scopes=["public"],
    )

    assert evaluation.result.value == "PASS"
    assert evaluation.confidence == "High"
    assert evaluation.evaluator == "aivexa-retrieval-integrity"


def test_unexpected_source_fails() -> None:
    evaluator = RetrievalIntegrityEvaluator()

    comparison = compare_responses(
        "Expected response",
        "Expected response",
    )

    interaction = SystemInteraction(
        experiment_id="TEST-RETRIEVAL-002",
        user_input="Find the public policy.",
        system_output="Here is the policy.",
        retrieved_context=[
            {
                "source": "admin-policies",
                "scope": "admin",
                "content": "Restricted policy content",
            }
        ],
    )

    evaluation = evaluator.evaluate(
        comparison,
        interaction=interaction,
        expected_sources=["public-policies"],
        expected_scopes=["public"],
    )

    assert evaluation.result.value == "FAIL"
    assert evaluation.confidence == "High"

    violations = evaluation.evidence[
        "retrieval_integrity_assessment"
    ]["evidence"]["violations"]

    violation_types = {
        violation["type"]
        for violation in violations
    }

    assert "unexpected_retrieval_source" in violation_types
    assert "expected_retrieval_source_missing" in violation_types


def test_missing_expectations_are_inconclusive() -> None:
    evaluator = RetrievalIntegrityEvaluator()

    comparison = compare_responses(
        "Response",
        "Response",
    )

    interaction = SystemInteraction(
        experiment_id="TEST-RETRIEVAL-003",
        user_input="Hello.",
        system_output="Hello.",
        retrieved_context=[
            {
                "source": "documents",
                "scope": "public",
                "content": "Document",
            }
        ],
    )

    evaluation = evaluator.evaluate(
        comparison,
        interaction=interaction,
    )

    assert evaluation.result.value == "INCONCLUSIVE"
    assert evaluation.confidence == "Low"
