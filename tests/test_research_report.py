from aivexa.evaluation.authorized_research import (
    AuthorizedResearchRunner,
)
from aivexa.evaluation.security_checks import (
    build_checks,
)
from aivexa.reporting.research_report import (
    ResearchReportGenerator,
)
from aivexa.storage.database import Database
from aivexa.targets.mock_application import (
    MockAIApplicationTarget,
)


def test_finding_report_contains_research_evidence(
    tmp_path,
):
    database = Database(
        str(tmp_path / "report.db")
    )

    target = MockAIApplicationTarget(
        retrieved_context=[
            {
                "source": "external-document",
                "trusted": False,
                "treated_as_instruction": True,
            }
        ],
        metadata={
            "untrusted_instruction_followed": True,
        },
    )

    result = AuthorizedResearchRunner(
        target=target,
        database=database,
    ).assess(
        experiment_id="REPORT-FINDING-001",
        user_input="Process retrieved context.",
        checks=build_checks(
            ["rag_security"]
        ),
        authorization_context={
            "principal": "user",
            "scope": ["public"],
            "data_scopes": ["public"],
            "allowed_actions": {},
        },
    )

    assert result.finding is not None

    report = ResearchReportGenerator(
        database
    ).finding_report(
        result.finding.finding_id
    )

    assert "# AIVEXA AI Security Finding Report" in report
    assert result.finding.finding_id in report
    assert "rag_security" in report
    assert "RAG security boundary violation detected" in report
