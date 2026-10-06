from aivexa.evaluation.action_boundary import (
    ActionBoundaryEvaluator,
)
from aivexa.evaluation.authorization import (
    AuthorizationBoundaryEvaluator,
)
from aivexa.evaluation.data_boundary import (
    DataBoundaryEvaluator,
)
from aivexa.evaluation.identity_boundary import (
    IdentityBoundaryEvaluator,
)
from aivexa.evaluation.retrieval_integrity import (
    RetrievalIntegrityEvaluator,
)
from aivexa.evaluation.tool_invocation import (
    ToolInvocationIntegrityEvaluator,
)
from aivexa.evaluation.agent_assessment import (
    AgentAssessmentEngine,
    AssessmentCheck,
)
from aivexa.evaluation.models import ExperimentResult
from aivexa.evidence.system import SystemInteraction


def test_phase4_complete_security_assessment():
    interaction = SystemInteraction(
        experiment_id="AVX-P4-E2E-001",
        user_input="search public information",
        system_output="Public information returned.",
        context={},
        retrieved_context=[
            {
                "source": "public-documents",
                "scope": "public",
            }
        ],
        tool_calls=[
            {
                "tool": "search",
                "arguments": {
                    "scope": "public",
                },
            }
        ],
        authorization_context={
            "principal": "user-1",
            "tenant": "tenant-a",
            "tools": ["search"],
            "scope": ["public"],
            "data_sources": ["public-documents"],
            "data_scopes": ["public"],
            "allowed_actions": {
                "search": {
                    "scope": "public",
                }
            },
        },
        metadata={
            "execution_identity": {
                "principal": "user-1",
                "tenant": "tenant-a",
            },
            "state_changes": [],
            "tool_results": [
                {
                    "tool": "search",
                    "trusted": True,
                }
            ],
        },
    )

    checks = [
        AssessmentCheck(
            name="authorization",
            evaluator=AuthorizationBoundaryEvaluator(),
        ),
        AssessmentCheck(
            name="identity",
            evaluator=IdentityBoundaryEvaluator(),
            kwargs={
                "expected_principal": "user-1",
                "expected_tenant": "tenant-a",
            },
        ),
        AssessmentCheck(
            name="data",
            evaluator=DataBoundaryEvaluator(),
        ),
        AssessmentCheck(
            name="action",
            evaluator=ActionBoundaryEvaluator(),
        ),
        AssessmentCheck(
            name="retrieval",
            evaluator=RetrievalIntegrityEvaluator(),
            kwargs={
                "expected_sources": [
                    "public-documents"
                ],
                "expected_scopes": ["public"],
            },
        ),
        AssessmentCheck(
            name="tool_invocation",
            evaluator=ToolInvocationIntegrityEvaluator(),
            kwargs={
                "expected_tool": "search",
                "expected_arguments": {
                    "scope": "public"
                },
            },
        ),
    ]

    assessment = AgentAssessmentEngine().assess(
        assessment_id="AVX-P4-ASSESSMENT-E2E-001",
        interaction=interaction,
        checks=checks,
    )

    assert assessment.result == ExperimentResult.PASS
    assert assessment.confidence == "High"
    assert len(assessment.evaluations) == 6

    assert all(
        evaluation.result == ExperimentResult.PASS
        for evaluation in assessment.evaluations
    )

    assert len(assessment.evidence["evaluations"]) == 6
    assert assessment.evidence["interaction"][
        "experiment_id"
    ] == "AVX-P4-E2E-001"
