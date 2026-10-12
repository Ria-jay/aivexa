from __future__ import annotations

from aivexa.comparison.behavior import compare_responses
from aivexa.evaluation.groundedness import GroundednessEvaluator
from aivexa.evaluation.models import ExperimentResult
from aivexa.evidence.system import SystemInteraction


def _interaction(system_output, retrieved_context, metadata=None):
    return SystemInteraction(
        experiment_id="GROUND-001",
        user_input="What does the document say?",
        system_output=system_output,
        retrieved_context=retrieved_context,
        metadata=metadata or {},
    )


def _evaluate(interaction):
    return GroundednessEvaluator().evaluate(
        compare_responses(
            interaction.system_output,
            interaction.system_output,
        ),
        interaction=interaction,
    )


def test_no_retrieval_is_not_applicable():
    result = _evaluate(_interaction("hello", []))

    assert result.result == ExperimentResult.NOT_APPLICABLE


def test_supported_response_passes():
    result = _evaluate(
        _interaction(
            "The password manager securely stores credentials.",
            [
                {
                    "content": (
                        "A password manager securely stores "
                        "credentials for accounts."
                    )
                }
            ],
        )
    )

    assert result.result == ExperimentResult.PASS
    assert result.evidence["support_ratio"] >= 0.5


def test_unsupported_response_is_anomaly():
    result = _evaluate(
        _interaction(
            "The reactor core temperature exceeded the limit.",
            [
                {
                    "content": (
                        "A password manager securely stores "
                        "credentials."
                    )
                }
            ],
        )
    )

    assert result.result == ExperimentResult.ANOMALY


def test_explicit_unsupported_claims_fail():
    result = _evaluate(
        _interaction(
            "Some output.",
            [{"content": "Some output."}],
            metadata={"unsupported_claims": ["invented claim"]},
        )
    )

    assert result.result == ExperimentResult.FAIL


def test_explicit_grounded_false_fails():
    result = _evaluate(
        _interaction(
            "Some output.",
            [{"content": "Some output."}],
            metadata={"grounded": False},
        )
    )

    assert result.result == ExperimentResult.FAIL
