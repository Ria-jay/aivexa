import json

from aivexa.evaluation.models import ExperimentResult
from aivexa.evaluation.finding import Finding
from aivexa.evaluation.reproduction import ReproductionBuilder
from aivexa.storage.database import Database


def test_reproduction_persists_with_lineage(tmp_path) -> None:
    database = Database(
        str(tmp_path / "reproduction.db")
    )

    finding = Finding(
        finding_id="FINDING-PERSIST-REPRO-001",
        assessment_id="ASSESSMENT-PERSIST-REPRO-001",
        experiment_id="EXPERIMENT-PERSIST-REPRO-001",
        title="Agent security boundary violation",
        result=ExperimentResult.FAIL,
        confidence="High",
        affected_properties=[
            "authorization_boundary_preservation",
        ],
        rationale="Unauthorized behavior observed.",
        evidence={},
    )

    builder = ReproductionBuilder()

    reproduction = builder.create(
        reproduction_id="REPRO-PERSIST-001",
        finding=finding,
        follow_up_experiment_id=(
            "EXPERIMENT-PERSIST-REPRO-FOLLOWUP-001"
        ),
    )

    reproduction = builder.resolve(
        reproduction=reproduction,
        result=ExperimentResult.FAIL,
        confidence="High",
        rationale="Violation reproduced.",
        evidence={
            "repeat": True,
        },
    )

    database.save_reproduction(reproduction)

    row = database.connection.execute(
        """
        SELECT
            reproduction_id,
            finding_id,
            source_experiment_id,
            follow_up_experiment_id,
            status,
            result,
            confidence,
            evidence
        FROM reproductions
        WHERE reproduction_id = ?
        """,
        ("REPRO-PERSIST-001",),
    ).fetchone()

    assert row is not None
    assert row[0] == "REPRO-PERSIST-001"
    assert row[1] == "FINDING-PERSIST-REPRO-001"
    assert row[2] == "EXPERIMENT-PERSIST-REPRO-001"
    assert (
        row[3]
        == "EXPERIMENT-PERSIST-REPRO-FOLLOWUP-001"
    )
    assert row[4] == "REPRODUCED"
    assert row[5] == "FAIL"
    assert row[6] == "High"

    evidence = json.loads(row[7])

    assert evidence["source_finding"] == (
        "FINDING-PERSIST-REPRO-001"
    )
    assert evidence["follow_up_result"]["result"] == "FAIL"
    assert evidence["follow_up_result"]["evidence"]["repeat"] is True
