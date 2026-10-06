from aivexa.research.reliability import (
    EvaluatorReliabilityCalculator,
)
from aivexa.storage.database import Database


def seed(
    database: Database,
    *,
    experiment_id: str,
    target: str = "model-a",
    result: str = "PASS",
    input_data: str = "same controlled input",
    evaluator: str = "aivexa-authorization-boundary",
    property_name: str = "authorization_boundary_preservation",
) -> None:
    database.save_experiment(
        {
            "experiment_id": experiment_id,
            "target": target,
            "objective": "Authorization reliability study",
            "hypothesis": (
                "The authorization boundary remains preserved."
            ),
            "intervention": "AUTHORIZATION_PROBE",
            "context": {
                "phase": 5,
                "reliability_test": True,
            },
            "created_at": (
                f"2026-01-01T00:00:0{experiment_id[-1]}+00:00"
            ),
            "input_data": input_data,
            "output_data": "controlled output",
            "result": result,
            "rationale": "Reliability test.",
            "confidence": "High",
            "observations": [
                "Controlled observation."
            ],
            "property_name": property_name,
            "property_expectation": (
                "Authorization boundary remains preserved."
            ),
            "comparison_group": None,
            "evaluator": evaluator,
            "reproducibility": {
                "seed": 42,
            },
        }
    )


def test_reliability_detects_consistent_repeated_results(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        result="PASS",
    )

    seed(
        database,
        experiment_id="EXP-002",
        result="PASS",
    )

    seed(
        database,
        experiment_id="EXP-003",
        result="PASS",
    )

    calculator = EvaluatorReliabilityCalculator(database)

    results = calculator.calculate()

    assert len(results) == 1

    reliability = results[0]

    assert reliability.evaluator == (
        "aivexa-authorization-boundary"
    )
    assert reliability.total_experiments == 3
    assert reliability.repeated_groups == 1
    assert reliability.consistent_groups == 1
    assert reliability.inconsistent_groups == 0
    assert reliability.agreement_rate == 1.0

    group = reliability.groups[0]

    assert group.experiment_count == 3
    assert group.result_counts == {
        "PASS": 3
    }
    assert group.agreement == 1.0
    assert group.consistent is True


def test_reliability_detects_inconsistent_repeated_results(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        result="PASS",
    )

    seed(
        database,
        experiment_id="EXP-002",
        result="FAIL",
    )

    seed(
        database,
        experiment_id="EXP-003",
        result="PASS",
    )

    calculator = EvaluatorReliabilityCalculator(database)

    results = calculator.calculate()

    assert len(results) == 1

    reliability = results[0]

    assert reliability.repeated_groups == 1
    assert reliability.consistent_groups == 0
    assert reliability.inconsistent_groups == 1
    assert reliability.agreement_rate == 0.0

    group = reliability.groups[0]

    assert group.result_counts == {
        "FAIL": 1,
        "PASS": 2,
    }
    assert group.agreement == 2 / 3
    assert group.consistent is False


def test_reliability_does_not_compare_different_inputs(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        input_data="input one",
    )

    seed(
        database,
        experiment_id="EXP-002",
        input_data="input two",
    )

    calculator = EvaluatorReliabilityCalculator(database)

    results = calculator.calculate()

    assert len(results) == 1

    reliability = results[0]

    assert reliability.total_experiments == 2
    assert reliability.repeated_groups == 0
    assert reliability.consistent_groups == 0
    assert reliability.inconsistent_groups == 0
    assert reliability.agreement_rate == 0.0
    assert reliability.groups == []


def test_reliability_separates_evaluators(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        evaluator="aivexa-authorization-boundary",
    )

    seed(
        database,
        experiment_id="EXP-002",
        evaluator="aivexa-authorization-boundary",
    )

    seed(
        database,
        experiment_id="EXP-003",
        evaluator="aivexa-data-boundary",
        property_name="data_boundary_preservation",
    )

    seed(
        database,
        experiment_id="EXP-004",
        evaluator="aivexa-data-boundary",
        property_name="data_boundary_preservation",
    )

    calculator = EvaluatorReliabilityCalculator(database)

    results = calculator.calculate()

    assert [
        item.evaluator
        for item in results
    ] == [
        "aivexa-authorization-boundary",
        "aivexa-data-boundary",
    ]

    assert all(
        item.repeated_groups == 1
        for item in results
    )


def test_reliability_filters_evaluator_and_target(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-a",
    )

    seed(
        database,
        experiment_id="EXP-003",
        target="model-b",
    )

    calculator = EvaluatorReliabilityCalculator(database)

    results = calculator.calculate(
        evaluator="aivexa-authorization-boundary",
        target="model-a",
    )

    assert len(results) == 1
    assert results[0].total_experiments == 2
    assert results[0].repeated_groups == 1
