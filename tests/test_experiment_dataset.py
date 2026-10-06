import json

from aivexa.research.experiment_dataset import ExperimentDataset
from aivexa.storage.database import Database


def seed_experiment(
    database: Database,
    *,
    experiment_id: str,
    target: str = "test-target",
    result: str = "PASS",
    confidence: str = "High",
    property_name: str = "authorization_boundary_preservation",
    evaluator: str = "aivexa-authorization-boundary",
) -> None:
    database.save_experiment(
        {
            "experiment_id": experiment_id,
            "target": target,
            "objective": "Test experiment",
            "hypothesis": "The security boundary remains preserved.",
            "intervention": "CONTROLLED_PROBE",
            "context": {
                "phase": 5,
                "test": True,
            },
            "created_at": f"2026-01-01T00:00:0{experiment_id[-1]}+00:00",
            "input_data": "test input",
            "output_data": "test output",
            "result": result,
            "rationale": "Controlled test result.",
            "confidence": confidence,
            "observations": [
                "Observation one.",
                "Observation two.",
            ],
            "property_name": property_name,
            "property_expectation": (
                "The declared security boundary remains preserved."
            ),
            "comparison_group": None,
            "evaluator": evaluator,
            "reproducibility": {
                "seed": 42,
                "temperature": 0,
            },
        }
    )


def test_dataset_row_contains_flattened_experiment_data(tmp_path):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
    )

    dataset = ExperimentDataset(database)
    rows = dataset.rows()

    assert len(rows) == 1

    row = rows[0]

    assert row.schema_version == "1.0"
    assert row.experiment_id == "EXP-001"
    assert row.target == "test-target"
    assert row.result == "PASS"
    assert row.confidence == "High"
    assert row.property_name == (
        "authorization_boundary_preservation"
    )
    assert row.observation_count == 2
    assert row.context["phase"] == 5
    assert row.reproducibility["seed"] == 42


def test_dataset_filters_by_result_and_property(tmp_path):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
        result="PASS",
    )

    seed_experiment(
        database,
        experiment_id="EXP-002",
        result="FAIL",
        property_name="data_boundary_preservation",
        evaluator="aivexa-data-boundary",
    )

    dataset = ExperimentDataset(database)

    failures = dataset.rows(result="FAIL")

    assert [row.experiment_id for row in failures] == [
        "EXP-002"
    ]

    data_boundary = dataset.rows(
        property_name="data_boundary_preservation"
    )

    assert [row.experiment_id for row in data_boundary] == [
        "EXP-002"
    ]


def test_dataset_exports_json_jsonl_and_csv(tmp_path):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
    )

    dataset = ExperimentDataset(database)

    json_path = dataset.export_json(
        tmp_path / "experiments.json"
    )

    jsonl_path = dataset.export_jsonl(
        tmp_path / "experiments.jsonl"
    )

    csv_path = dataset.export_csv(
        tmp_path / "experiments.csv"
    )

    assert json_path.exists()
    assert jsonl_path.exists()
    assert csv_path.exists()

    json_payload = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    assert json_payload["dataset_type"] == (
        "aivexa-experiment-dataset"
    )
    assert json_payload["row_count"] == 1
    assert json_payload["rows"][0]["experiment_id"] == (
        "EXP-001"
    )

    jsonl_lines = [
        line
        for line in jsonl_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]

    assert len(jsonl_lines) == 1

    jsonl_record = json.loads(jsonl_lines[0])

    assert jsonl_record["experiment_id"] == "EXP-001"

    csv_text = csv_path.read_text(encoding="utf-8")

    assert "experiment_id" in csv_text
    assert "EXP-001" in csv_text
    assert "authorization_boundary_preservation" in csv_text


def test_dataset_order_is_deterministic(tmp_path):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-002",
    )

    seed_experiment(
        database,
        experiment_id="EXP-001",
    )

    dataset = ExperimentDataset(database)

    first = [
        row.experiment_id
        for row in dataset.rows()
    ]

    second = [
        row.experiment_id
        for row in dataset.rows()
    ]

    assert first == second
    assert first == [
        "EXP-001",
        "EXP-002",
    ]
