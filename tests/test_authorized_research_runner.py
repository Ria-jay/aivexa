from aivexa.evaluation.authorized_research import (
    AuthorizedResearchRunner,
)
from aivexa.evaluation.security_checks import (
    build_checks,
)
from aivexa.storage.database import Database
from aivexa.targets.mock_application import (
    MockAIApplicationTarget,
)


def test_authorized_research_runner_persists_assessment(
    tmp_path,
):
    database = Database(
        str(tmp_path / "research.db")
    )

    runner = AuthorizedResearchRunner(
        target=MockAIApplicationTarget(),
        database=database,
    )

    result = runner.assess(
        experiment_id="AUTHORIZED-001",
        user_input="Find public information.",
        checks=build_checks(
            [
                "authorization",
                "data_boundary",
                "action_boundary",
                "rag_security",
                "context_boundary",
                "tool_invocation",
                "tool_result_boundary",
                "side_effect",
                "identity_boundary",
                "output_handling",
            ]
        ),
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
        "AUTHORIZED-001-ASSESSMENT"
    )

    row = database.connection.execute(
        """
        SELECT assessment_id, assessment_result
        FROM experiments
        WHERE experiment_id = ?
        """,
        ("AUTHORIZED-001",),
    ).fetchone()

    assert row[0] == "AUTHORIZED-001-ASSESSMENT"
    assert row[1] is not None


def test_research_runner_creates_investigation_on_failure(
    tmp_path,
):
    database = Database(
        str(tmp_path / "research-failure.db")
    )

    target = MockAIApplicationTarget(
        retrieved_context=[
            {
                "source": "external-document",
                "trusted": False,
                "treated_as_instruction": True,
                "caused_action": True,
            }
        ],
        metadata={
            "untrusted_instruction_followed": True,
        },
    )

    runner = AuthorizedResearchRunner(
        target=target,
        database=database,
    )

    result = runner.assess(
        experiment_id="AUTHORIZED-FAIL-001",
        user_input="Use the retrieved information.",
        checks=build_checks(
            [
                "rag_security",
            ]
        ),
        authorization_context={
            "principal": "user",
            "scope": ["public"],
            "data_scopes": ["public"],
            "allowed_actions": {},
        },
    )

    assert result.assessment.result.value == "FAIL"
    assert result.finding is not None
    assert result.follow_up is not None
    assert result.investigation is not None

    finding = database.connection.execute(
        """
        SELECT finding_id
        FROM findings
        WHERE experiment_id = ?
        """,
        ("AUTHORIZED-FAIL-001",),
    ).fetchone()

    assert finding is not None
