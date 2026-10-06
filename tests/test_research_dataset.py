import json

from aivexa.evaluation.models import ExperimentResult
from aivexa.research.dataset import (
    ResearchDataset,
    build_research_record,
    calculate_fingerprint,
)
from aivexa.storage.database import Database


def _save_experiment(
    database,
    experiment_id="RESEARCH-DATASET-001",
):
    database.save_experiment(
        {
            "experiment_id": experiment_id,
            "target": "research-test-target",
            "objective": "Measure controlled behavior.",
            "hypothesis": "The declared boundary will hold.",
            "intervention": "CONTROLLED_PROBE",
            "context": {
                "test": True,
                "scope": "authorized",
            },
            "created_at": "2026-10-01T12:00:00+00:00",
            "input_data": "controlled input",
            "output_data": "controlled output",
            "result": ExperimentResult.FAIL.value,
            "rationale": "A controlled boundary failure occurred.",
            "confidence": "High",
            "observations": [
                "Observed controlled behavior."
            ],
            "property_name": "test_boundary",
            "property_expectation": (
                "The boundary remains preserved."
            ),
            "comparison_group": None,
            "evaluator": "test-evaluator",
            "reproducibility": {
                "model": "test-model",
                "model_version": "test-version",
                "configuration": {
                    "temperature": 0,
                },
                "evaluation_policy": "test-policy",
            },
        }
    )


def test_research_record_contains_full_lineage(tmp_path):
    database = Database(
        str(tmp_path / "dataset.db")
    )

    _save_experiment(database)

    database.connection.execute(
        """
        UPDATE experiments
        SET
            assessment_id = ?,
            assessment_result = ?,
            assessment_confidence = ?,
            assessment_rationale = ?,
            assessment_evidence = ?
        WHERE experiment_id = ?
        """,
        (
            "ASSESSMENT-001",
            "FAIL",
            "High",
            "Assessment failed.",
            json.dumps(
                {
                    "individual_evaluations": [
                        {
                            "property_name": "test_boundary",
                            "result": "FAIL",
                        }
                    ]
                }
            ),
            "RESEARCH-DATASET-001",
        ),
    )

    database.connection.commit()

    record = build_research_record(
        database,
        "RESEARCH-DATASET-001",
    )

    assert record.schema_version == "1.0"
    assert record.experiment_id == "RESEARCH-DATASET-001"
    assert record.target == "research-test-target"
    assert record.assessment is not None
    assert record.reproducibility["model"] == "test-model"
    assert record.reproducibility["model_version"] == "test-version"
    assert len(record.fingerprint) == 64


def test_research_fingerprint_is_deterministic():
    experiment = {
        "experiment_id": "X",
        "input_data": "hello",
    }

    assessment = {
        "result": "PASS",
    }

    reproducibility = {
        "model": "model-a",
        "temperature": 0,
    }

    first = calculate_fingerprint(
        experiment,
        assessment,
        None,
        [],
        reproducibility,
    )

    second = calculate_fingerprint(
        experiment,
        assessment,
        None,
        [],
        reproducibility,
    )

    assert first == second
    assert len(first) == 64


def test_dataset_exports_json_and_jsonl(tmp_path):
    database = Database(
        str(tmp_path / "dataset.db")
    )

    _save_experiment(database)

    dataset = ResearchDataset(database)

    json_path = dataset.export_json(
        tmp_path / "research.json"
    )
    jsonl_path = dataset.export_jsonl(
        tmp_path / "research.jsonl"
    )

    assert json_path.exists()
    assert jsonl_path.exists()

    payload = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == "1.0"
    assert payload["dataset_type"] == (
        "aivexa-research-dataset"
    )
    assert len(payload["records"]) == 1

    lines = [
        line
        for line in jsonl_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["experiment_id"] == (
        "RESEARCH-DATASET-001"
    )


def test_existing_database_migrates_reproducibility_column(
    tmp_path,
):
    path = tmp_path / "legacy.db"

    database = Database(str(path))

    columns = {
        row[1]
        for row in database.connection.execute(
            "PRAGMA table_info(experiments)"
        )
    }

    assert "reproducibility" in columns
