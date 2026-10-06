from aivexa.research.coverage import CoverageCalculator
from aivexa.storage.database import Database


def seed(
    database: Database,
    *,
    experiment_id: str,
    target: str,
    objective: str,
    intervention: str,
    result: str,
    property_name: str,
    evaluator: str,
) -> None:
    database.save_experiment(
        {
            "experiment_id": experiment_id,
            "target": target,
            "objective": objective,
            "hypothesis": "Declared property remains preserved.",
            "intervention": intervention,
            "context": {
                "phase": 5,
            },
            "created_at": (
                f"2026-01-01T00:00:0{experiment_id[-1]}+00:00"
            ),
            "input_data": "controlled input",
            "output_data": "controlled output",
            "result": result,
            "rationale": "Coverage test.",
            "confidence": "High",
            "observations": [
                "Controlled observation."
            ],
            "property_name": property_name,
            "property_expectation": "Property remains preserved.",
            "comparison_group": None,
            "evaluator": evaluator,
            "reproducibility": {
                "seed": 42,
            },
        }
    )


def test_coverage_counts_distinct_research_dimensions(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
        objective="Authorization evaluation",
        intervention="AUTHORIZATION_PROBE",
        result="PASS",
        property_name="authorization_boundary_preservation",
        evaluator="aivexa-authorization-boundary",
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-b",
        objective="Data evaluation",
        intervention="DATA_BOUNDARY_PROBE",
        result="FAIL",
        property_name="data_boundary_preservation",
        evaluator="aivexa-data-boundary",
    )

    seed(
        database,
        experiment_id="EXP-003",
        target="model-a",
        objective="Authorization evaluation",
        intervention="AUTHORIZATION_PROBE",
        result="INCONCLUSIVE",
        property_name="authorization_boundary_preservation",
        evaluator="aivexa-authorization-boundary",
    )

    calculator = CoverageCalculator(database)

    coverage = calculator.calculate()

    assert coverage.total_experiments == 3

    assert coverage.target_coverage.covered == 2
    assert coverage.property_coverage.covered == 2
    assert coverage.evaluator_coverage.covered == 2
    assert coverage.result_coverage.covered == 3
    assert coverage.intervention_coverage.covered == 2
    assert coverage.objective_coverage.covered == 2

    assert coverage.targets == [
        "model-a",
        "model-b",
    ]

    assert coverage.properties == [
        "authorization_boundary_preservation",
        "data_boundary_preservation",
    ]


def test_coverage_proportions_are_deterministic(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
        objective="Authorization evaluation",
        intervention="AUTHORIZATION_PROBE",
        result="PASS",
        property_name="authorization_boundary_preservation",
        evaluator="aivexa-authorization-boundary",
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-a",
        objective="Authorization evaluation",
        intervention="AUTHORIZATION_PROBE",
        result="PASS",
        property_name="authorization_boundary_preservation",
        evaluator="aivexa-authorization-boundary",
    )

    calculator = CoverageCalculator(database)

    coverage = calculator.calculate()

    assert coverage.target_coverage.proportion == 0.5
    assert coverage.property_coverage.proportion == 0.5
    assert coverage.evaluator_coverage.proportion == 0.5
    assert coverage.result_coverage.proportion == 0.5
    assert coverage.intervention_coverage.proportion == 0.5
    assert coverage.objective_coverage.proportion == 0.5


def test_coverage_supports_filters(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
        objective="Authorization evaluation",
        intervention="AUTHORIZATION_PROBE",
        result="PASS",
        property_name="authorization_boundary_preservation",
        evaluator="aivexa-authorization-boundary",
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-b",
        objective="Data evaluation",
        intervention="DATA_BOUNDARY_PROBE",
        result="FAIL",
        property_name="data_boundary_preservation",
        evaluator="aivexa-data-boundary",
    )

    calculator = CoverageCalculator(database)

    coverage = calculator.calculate(
        target="model-a",
    )

    assert coverage.total_experiments == 1
    assert coverage.targets == ["model-a"]
    assert coverage.properties == [
        "authorization_boundary_preservation"
    ]
    assert coverage.results == ["PASS"]


def test_empty_coverage_is_defined(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    calculator = CoverageCalculator(database)

    coverage = calculator.calculate()

    assert coverage.total_experiments == 0

    assert coverage.target_coverage.covered == 0
    assert coverage.target_coverage.total == 0
    assert coverage.target_coverage.proportion == 0.0

    assert coverage.property_coverage.covered == 0
    assert coverage.evaluator_coverage.covered == 0
    assert coverage.result_coverage.covered == 0
    assert coverage.intervention_coverage.covered == 0
    assert coverage.objective_coverage.covered == 0

    assert coverage.targets == []
    assert coverage.properties == []
    assert coverage.evaluators == []
    assert coverage.results == []
    assert coverage.interventions == []
    assert coverage.objectives == []
