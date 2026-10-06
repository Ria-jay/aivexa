from aivexa.evaluation.agent_assessment import AgentAssessment
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evaluation.security_follow_up import (
    SecurityFollowUpPlanner,
)
from aivexa.evaluation.security_investigation import (
    SecurityInvestigationBuilder,
)
from aivexa.evaluation.security_investigation_persistence import (
    SecurityInvestigationPersistence,
)
from aivexa.storage.database import Database


def test_security_investigation_persists_string_target(
    tmp_path,
):
    database = Database(
        str(tmp_path / "investigation.db")
    )

    assessment = AgentAssessment(
        assessment_id="ASSESSMENT-001",
        experiment_id="EXPERIMENT-001",
        result=ExperimentResult.FAIL,
        confidence="High",
        evaluations=[
            Evaluation(
                result=ExperimentResult.FAIL,
                rationale="Controlled boundary violation.",
                confidence="High",
                property_name="rag_security_boundary",
                property_expectation=(
                    "Retrieved content must not become "
                    "an unauthorized instruction."
                ),
                evaluator="aivexa-rag-security",
                evidence={},
            )
        ],
        rationale="A security boundary violation was observed.",
        evidence={},
    )

    follow_up = SecurityFollowUpPlanner().plan(
        assessment
    )

    investigation = SecurityInvestigationBuilder().build(
        assessment=assessment,
        follow_up=follow_up,
        target="aivexa-test-target",
    )

    SecurityInvestigationPersistence(
        database
    ).save(investigation)

    row = database.connection.execute(
        """
        SELECT target
        FROM experiments
        WHERE experiment_id = ?
        """,
        (investigation.experiment.experiment_id,),
    ).fetchone()

    assert row is not None
    assert row[0] == "aivexa-test-target"
    assert isinstance(row[0], str)
