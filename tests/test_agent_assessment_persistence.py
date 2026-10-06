import json
import sqlite3

from aivexa.evaluation.agent_assessment import AgentAssessment
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.storage.database import Database


def test_agent_assessment_persists_against_experiment(
    tmp_path,
) -> None:
    database_path = tmp_path / "assessment.db"

    database = Database(str(database_path))

    database.save_experiment(
        {
            "experiment_id": "AVX-P4-PERSIST-001",
            "target": "test-agent",
            "objective": "Test persistence",
            "hypothesis": "Assessment should persist.",
            "intervention": "CONTROLLED_TEST",
            "context": {},
            "created_at": "2026-10-01T00:00:00+00:00",
            "input_data": "test input",
            "output_data": "test output",
            "result": "PASS",
            "rationale": "Initial experiment.",
            "confidence": "High",
            "observations": [],
        }
    )

    evaluation = Evaluation(
        result=ExperimentResult.PASS,
        rationale="Authorization boundary preserved.",
        confidence="High",
        property_name="authorization_boundary_preservation",
        property_expectation="Stay within authorization.",
        evaluator="aivexa-authorization-boundary",
        evidence={
            "violations": [],
        },
    )

    assessment = AgentAssessment(
        assessment_id="AVX-P4-ASSESSMENT-PERSIST-001",
        experiment_id="AVX-P4-PERSIST-001",
        result=ExperimentResult.PASS,
        confidence="High",
        evaluations=[evaluation],
        rationale="All evaluated properties passed.",
        evidence={
            "engine": "aivexa-agent-assessment",
        },
    )

    database.save_assessment(assessment)

    row = database.connection.execute(
        """
        SELECT
            assessment_id,
            assessment_result,
            assessment_confidence,
            assessment_rationale,
            assessment_evidence
        FROM experiments
        WHERE experiment_id = ?
        """,
        ("AVX-P4-PERSIST-001",),
    ).fetchone()

    assert row is not None

    assert row[0] == "AVX-P4-ASSESSMENT-PERSIST-001"
    assert row[1] == "PASS"
    assert row[2] == "High"
    assert row[3] == "All evaluated properties passed."

    evidence = json.loads(row[4])

    assert evidence["engine"] == "aivexa-agent-assessment"
    assert len(evidence["individual_evaluations"]) == 1

    individual = evidence["individual_evaluations"][0]

    assert (
        individual["property_name"]
        == "authorization_boundary_preservation"
    )
    assert individual["result"] == "PASS"
    assert individual["evaluator"] == (
        "aivexa-authorization-boundary"
    )


def test_existing_database_migrates_assessment_columns(
    tmp_path,
) -> None:
    database_path = tmp_path / "migration.db"

    connection = sqlite3.connect(database_path)

    connection.execute(
        """
        CREATE TABLE experiments (
            experiment_id TEXT PRIMARY KEY,
            target TEXT NOT NULL,
            objective TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            intervention TEXT NOT NULL,
            context TEXT NOT NULL,
            created_at TEXT NOT NULL,
            input_data TEXT NOT NULL,
            output_data TEXT NOT NULL,
            result TEXT NOT NULL,
            rationale TEXT NOT NULL,
            confidence TEXT NOT NULL,
            observations TEXT NOT NULL,
            property_name TEXT,
            property_expectation TEXT,
            comparison_group TEXT,
            evaluator TEXT
        )
        """
    )

    connection.commit()
    connection.close()

    database = Database(str(database_path))

    columns = {
        row[1]
        for row in database.connection.execute(
            "PRAGMA table_info(experiments)"
        )
    }

    assert "assessment_id" in columns
    assert "assessment_result" in columns
    assert "assessment_confidence" in columns
    assert "assessment_rationale" in columns
    assert "assessment_evidence" in columns
