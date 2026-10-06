import json

import pytest

from aivexa.research.benchmark import (
    BenchmarkBuilder,
    BenchmarkCase,
    BenchmarkExecutionEngine,
    load_benchmark,
    save_benchmark,
)
from aivexa.storage.database import Database


def make_case(
    case_id: str,
    *,
    input_data: str = "controlled input",
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        name=f"Case {case_id}",
        domain="security",
        objective="Evaluate authorization preservation",
        hypothesis=(
            "The target preserves the declared authorization boundary."
        ),
        intervention="AUTHORIZATION_PROBE",
        property_name="authorization_boundary_preservation",
        property_expectation=(
            "Authorization boundary remains preserved."
        ),
        input_data=input_data,
        metadata={
            "source": "phase-5-benchmark-test",
        },
    )


def make_benchmark() -> object:
    return BenchmarkBuilder().build(
        benchmark_id="AIVEXA-SEC-001",
        name="Authorization Boundary Benchmark",
        version="1.0.0",
        description=(
            "Controlled benchmark for authorization boundary evaluation."
        ),
        cases=[
            make_case("AUTH-001"),
            make_case(
                "AUTH-002",
                input_data="second controlled input",
            ),
        ],
        metadata={
            "research_phase": 5,
        },
    )


def seed_experiment(
    database: Database,
    *,
    experiment_id: str,
    input_data: str,
) -> None:
    database.save_experiment(
        {
            "experiment_id": experiment_id,
            "target": "model-a",
            "objective": "Evaluate authorization preservation",
            "hypothesis": (
                "The target preserves the declared authorization boundary."
            ),
            "intervention": "AUTHORIZATION_PROBE",
            "context": {
                "phase": 5,
                "benchmark_test": True,
            },
            "created_at": (
                f"2026-01-01T00:00:0{experiment_id[-1]}+00:00"
            ),
            "input_data": input_data,
            "output_data": "controlled output",
            "result": "PASS",
            "rationale": "Benchmark execution test.",
            "confidence": "High",
            "observations": [
                "Controlled benchmark observation."
            ],
            "property_name": (
                "authorization_boundary_preservation"
            ),
            "property_expectation": (
                "Authorization boundary remains preserved."
            ),
            "comparison_group": None,
            "evaluator": (
                "aivexa-authorization-boundary"
            ),
            "reproducibility": {
                "seed": 42,
            },
        }
    )


def test_benchmark_builder_creates_versioned_definition():
    benchmark = make_benchmark()

    assert benchmark.benchmark_id == "AIVEXA-SEC-001"
    assert benchmark.version == "1.0.0"
    assert len(benchmark.cases) == 2
    assert benchmark.case("AUTH-001").input_data == (
        "controlled input"
    )


def test_benchmark_rejects_duplicate_case_ids():
    case = make_case("AUTH-001")

    with pytest.raises(ValueError, match="unique"):
        BenchmarkBuilder().build(
            benchmark_id="AIVEXA-SEC-001",
            name="Invalid benchmark",
            version="1.0.0",
            description="Invalid.",
            cases=[case, case],
        )


def test_benchmark_definition_round_trips_as_json(
    tmp_path,
):
    benchmark = make_benchmark()

    path = save_benchmark(
        benchmark,
        tmp_path / "benchmark.json",
    )

    loaded = load_benchmark(path)

    assert loaded.to_dict() == benchmark.to_dict()

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert payload["benchmark_id"] == (
        "AIVEXA-SEC-001"
    )
    assert payload["case_count"] == 2


def test_benchmark_execution_maps_recorded_experiments(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
        input_data="controlled input",
    )

    seed_experiment(
        database,
        experiment_id="EXP-002",
        input_data="second controlled input",
    )

    benchmark = make_benchmark()

    engine = BenchmarkExecutionEngine(database)

    executions = engine.execute_recorded(
        benchmark=benchmark,
        target="model-a",
    )

    assert len(executions) == 2

    assert [
        execution.case_id
        for execution in executions
    ] == [
        "AUTH-001",
        "AUTH-002",
    ]

    assert all(
        execution.benchmark_id == "AIVEXA-SEC-001"
        for execution in executions
    )


def test_benchmark_coverage_identifies_unexecuted_cases(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
        input_data="controlled input",
    )

    benchmark = make_benchmark()

    engine = BenchmarkExecutionEngine(database)

    coverage = engine.coverage(
        benchmark=benchmark,
        target="model-a",
    )

    assert coverage["total_cases"] == 2
    assert coverage["executed_cases"] == 1
    assert coverage["unexecuted_cases"] == [
        "AUTH-002"
    ]
    assert coverage["coverage"] == 0.5
