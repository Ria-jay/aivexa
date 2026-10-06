from __future__ import annotations

from aivexa.evaluation.action_boundary import (
    ActionBoundaryEvaluator,
)
from aivexa.evaluation.agent_assessment import AssessmentCheck
from aivexa.evaluation.authorization import (
    AuthorizationBoundaryEvaluator,
)
from aivexa.evaluation.context_boundary import (
    ContextBoundaryEvaluator,
)
from aivexa.evaluation.data_boundary import (
    DataBoundaryEvaluator,
)
from aivexa.evaluation.identity_boundary import (
    IdentityBoundaryEvaluator,
)
from aivexa.evaluation.output_handling import (
    OutputHandlingEvaluator,
)
from aivexa.evaluation.rag_security import (
    RAGSecurityEvaluator,
)
from aivexa.evaluation.retrieval_integrity import (
    RetrievalIntegrityEvaluator,
)
from aivexa.evaluation.side_effect import (
    SideEffectIntegrityEvaluator,
)
from aivexa.evaluation.tool_invocation import (
    ToolInvocationIntegrityEvaluator,
)
from aivexa.evaluation.tool_result_boundary import (
    ToolResultBoundaryEvaluator,
)


def build_default_security_checks() -> list[AssessmentCheck]:
    return [
        AssessmentCheck(
            name="authorization",
            evaluator=AuthorizationBoundaryEvaluator(),
        ),
        AssessmentCheck(
            name="data_boundary",
            evaluator=DataBoundaryEvaluator(),
        ),
        AssessmentCheck(
            name="action_boundary",
            evaluator=ActionBoundaryEvaluator(),
        ),
        AssessmentCheck(
            name="retrieval_integrity",
            evaluator=RetrievalIntegrityEvaluator(),
        ),
        AssessmentCheck(
            name="rag_security",
            evaluator=RAGSecurityEvaluator(),
        ),
        AssessmentCheck(
            name="context_boundary",
            evaluator=ContextBoundaryEvaluator(),
        ),
        AssessmentCheck(
            name="tool_invocation",
            evaluator=ToolInvocationIntegrityEvaluator(),
        ),
        AssessmentCheck(
            name="tool_result_boundary",
            evaluator=ToolResultBoundaryEvaluator(),
        ),
        AssessmentCheck(
            name="side_effect",
            evaluator=SideEffectIntegrityEvaluator(),
        ),
        AssessmentCheck(
            name="identity_boundary",
            evaluator=IdentityBoundaryEvaluator(),
        ),
        AssessmentCheck(
            name="output_handling",
            evaluator=OutputHandlingEvaluator(),
        ),
    ]


def build_checks(
    names: list[str] | None = None,
) -> list[AssessmentCheck]:
    checks = build_default_security_checks()

    if not names:
        return checks

    wanted = set(names)

    return [
        check
        for check in checks
        if check.name in wanted
    ]
