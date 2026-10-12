from __future__ import annotations

from aivexa.evaluation.agent_assessment import (
    AgentAssessmentEngine,
    AssessmentCheck,
)
from aivexa.evaluation.groundedness import GroundednessEvaluator
from aivexa.evaluation.confirmation import (
    ConfirmationBoundaryEvaluator,
)
from aivexa.evaluation.output_handling import (
    OutputHandlingEvaluator,
)
from aivexa.evidence.system import SystemInteraction


def test_not_applicable_checks_do_not_degrade_pass():
    interaction = SystemInteraction(
        experiment_id="NA-001",
        user_input="Summarize the document.",
        system_output="Public information returned.",
        retrieved_context=[],
        metadata={},
    )

    checks = [
        AssessmentCheck(
            name="output_handling",
            evaluator=OutputHandlingEvaluator(),
            kwargs={
                "required_markers": ["Public"],
            },
        ),
        AssessmentCheck(
            name="groundedness",
            evaluator=GroundednessEvaluator(),
        ),
        AssessmentCheck(
            name="confirmation",
            evaluator=ConfirmationBoundaryEvaluator(),
        ),
    ]

    assessment = AgentAssessmentEngine().assess(
        assessment_id="NA-ASSESSMENT-001",
        interaction=interaction,
        checks=checks,
    )

    # output_handling passes; groundedness and confirmation are
    # NOT_APPLICABLE and must not force INCONCLUSIVE.
    assert assessment.result.value == "PASS"
    assert assessment.confidence == "High"

    results = {
        evaluation.property_name: evaluation.result.value
        for evaluation in assessment.evaluations
    }

    assert results["output_handling_integrity"] == "PASS"
    assert results["response_groundedness"] == "NOT_APPLICABLE"
    assert (
        results["confirmation_boundary_preservation"]
        == "NOT_APPLICABLE"
    )


def test_all_not_applicable_is_not_applicable():
    interaction = SystemInteraction(
        experiment_id="NA-002",
        user_input="hello",
        system_output="hello",
    )

    checks = [
        AssessmentCheck(
            name="groundedness",
            evaluator=GroundednessEvaluator(),
        ),
        AssessmentCheck(
            name="confirmation",
            evaluator=ConfirmationBoundaryEvaluator(),
        ),
    ]

    assessment = AgentAssessmentEngine().assess(
        assessment_id="NA-ASSESSMENT-002",
        interaction=interaction,
        checks=checks,
    )

    assert assessment.result.value == "NOT_APPLICABLE"
