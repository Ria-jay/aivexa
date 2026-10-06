from aivexa.evaluation.agent_assessment import AssessmentCheck
from aivexa.evaluation.authorization import (
    AuthorizationBoundaryEvaluator,
)
from aivexa.evaluation.data_boundary import (
    DataBoundaryEvaluator,
)
from aivexa.evaluation.rag_security import (
    RAGSecurityEvaluator,
)
from aivexa.evaluation.research_workflow import (
    ResearchWorkflow,
)
from aivexa.storage.database import Database
from aivexa.targets.mock_application import (
    MockAIApplicationTarget,
)


def test_research_workflow_persists_assessment_and_investigation(
    tmp_path,
):
    database = Database(
        str(tmp_path / "workflow.db")
    )

    target = MockAIApplicationTarget()

    workflow = ResearchWorkflow(
        target=target,
        database=database,
    )

    result = workflow.assess(
        experiment_id="RESEARCH-001",
        user_input="Find public information.",
        checks=[
            AssessmentCheck(
                name="authorization",
                evaluator=AuthorizationBoundaryEvaluator(),
            ),
            AssessmentCheck(
                name="data",
                evaluator=DataBoundaryEvaluator(),
            ),
            AssessmentCheck(
                name="rag",
                evaluator=RAGSecurityEvaluator(),
            ),
        ],
        authorization_context={
            "principal": "user",
            "tenant": "tenant-a",
            "tools": [],
            "scope": ["public"],
            "data_sources": [],
            "data_scopes": ["public"],
            "allowed_actions": {},
        },
    )

    assert result.assessment.assessment_id == (
        "RESEARCH-001-ASSESSMENT"
    )

    row = database.connection.execute(
        """
        SELECT assessment_id
        FROM experiments
        WHERE experiment_id = ?
        """,
        ("RESEARCH-001",),
    ).fetchone()

    assert row[0] == "RESEARCH-001-ASSESSMENT"

    investigation = database.connection.execute(
        """
        SELECT experiment_id, comparison_group
        FROM experiments
        WHERE experiment_id LIKE ?
        """,
        ("RESEARCH-001-FOLLOWUP-%",),
    ).fetchone()

    assert investigation is not None
    assert investigation[1] == "RESEARCH-001"
