import json

from aivexa.evaluation.finding import Finding
from aivexa.evaluation.models import ExperimentResult
from aivexa.storage.database import Database


def test_finding_persists(tmp_path) -> None:
    database = Database(
        str(tmp_path / "finding.db")
    )

    finding = Finding(
        finding_id="FINDING-PERSIST-001",
        assessment_id="ASSESSMENT-PERSIST-001",
        experiment_id="EXPERIMENT-PERSIST-001",
        title="Agent security boundary violation",
        result=ExperimentResult.FAIL,
        confidence="High",
        affected_properties=[
            "authorization_boundary_preservation",
            "action_boundary_preservation",
        ],
        rationale="Unauthorized behavior was observed.",
        evidence={
            "individual_evaluations": [
                {
                    "property_name": (
                        "authorization_boundary_preservation"
                    ),
                    "result": "FAIL",
                }
            ]
        },
    )

    database.save_finding(finding)

    row = database.connection.execute(
        """
        SELECT
            finding_id,
            assessment_id,
            experiment_id,
            title,
            result,
            confidence,
            affected_properties,
            rationale,
            evidence
        FROM findings
        WHERE finding_id = ?
        """,
        ("FINDING-PERSIST-001",),
    ).fetchone()

    assert row is not None
    assert row[0] == "FINDING-PERSIST-001"
    assert row[1] == "ASSESSMENT-PERSIST-001"
    assert row[2] == "EXPERIMENT-PERSIST-001"
    assert row[3] == (
        "Agent security boundary violation"
    )
    assert row[4] == "FAIL"
    assert row[5] == "High"

    assert json.loads(row[6]) == [
        "authorization_boundary_preservation",
        "action_boundary_preservation",
    ]

    assert json.loads(row[8])[
        "individual_evaluations"
    ][0]["result"] == "FAIL"
