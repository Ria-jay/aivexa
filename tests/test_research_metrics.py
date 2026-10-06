from aivexa.research.metrics import ResearchMetricsCalculator
from aivexa.storage.database import Database


def seed(
    database: Database,
    *,
    experiment_id: str,
    target: str,
    result: str,
    confidence: str,
    property_name: str,
    evaluator: str,
    observations: list[str],
) -> None:
    database.save_experiment(
        {
            "experiment_id": experiment_id,
            "target": target,
            "objective": "Research metrics experiment",
            "hypothesis": "Declared property remains preserved.",
            "intervention": "CONTROLLED_PROBE",
            "context": {
                "phase": 5,
            },
            "created_at": (
                f"2026-01-01T00:00:0{experiment_id[-1]}+00:00"
            ),
            "input_data": "controlled input",
            "output_data": "controlled output",
            "result": result,
            "rationale": "Controlled research result.",
            "confidence": confidence,
            "observations": observations,
            "property_name": property_name,
            "property_expectation": "Property remains preserved.",
            "comparison_group": None,
            "evaluator": evaluator,
            "reproducibility": {
                "seed": 42,
            },
        }
    )


def test_metrics_calculate_dataset_distributions(tmp_path):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
        result="PASS",
        confidence="High",
        property_name="authorization_boundary_preservation",
        evaluator="aivexa-authorization-boundary",
        observations=["one", "two"],
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-b",
        result="FAIL",
        confidence="Medium",
        property_name="data_boundary_preservation",
        evaluator="aivexa-data-boundary",
        observations=["one"],
    )

    calculator = ResearchMetricsCalculator(database)

    metrics = calculator.calculate()

    assert metrics.total_experiments == 2
    assert metrics.unique_targets == 2
    assert metrics.unique_properties == 2
    assert metrics.unique_evaluators == 2

    assert metrics.result_distribution.counts == {
        "FAIL": 1,
        "PASS": 1,
    }

    assert metrics.result_distribution.proportions == {
        "FAIL": 0.5,
        "PASS": 0.5,
    }

    assert metrics.confidence_distribution.counts == {
        "High": 1,
        "Medium": 1,
    }


def test_metrics_calculate_observation_and_assessment_coverage(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
        result="PASS",
        confidence="High",
        property_name="authorization_boundary_preservation",
        evaluator="aivexa-authorization-boundary",
        observations=["one", "two"],
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-a",
        result="INCONCLUSIVE",
        confidence="Low",
        property_name="authorization_boundary_preservation",
        evaluator="aivexa-authorization-boundary",
        observations=[],
    )

    calculator = ResearchMetricsCalculator(database)

    metrics = calculator.calculate()

    assert metrics.observation_metrics[
        "total_observations"
    ] == 2.0

    assert metrics.observation_metrics[
        "average_observations_per_experiment"
    ] == 1.0

    assert metrics.observation_metrics[
        "experiments_with_observations"
    ] == 1.0

    assert metrics.observation_metrics[
        "observation_coverage"
    ] == 0.5

    assert metrics.observation_metrics[
        "experiments_with_assessments"
    ] == 0.0

    assert metrics.observation_metrics[
        "assessment_coverage"
    ] == 0.0


def test_metrics_support_dataset_filters(tmp_path):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
        result="PASS",
        confidence="High",
        property_name="authorization_boundary_preservation",
        evaluator="aivexa-authorization-boundary",
        observations=["one"],
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-b",
        result="FAIL",
        confidence="High",
        property_name="data_boundary_preservation",
        evaluator="aivexa-data-boundary",
        observations=["one"],
    )

    calculator = ResearchMetricsCalculator(database)

    metrics = calculator.calculate(
        target="model-a",
    )

    assert metrics.total_experiments == 1
    assert metrics.unique_targets == 1
    assert metrics.result_distribution.counts == {
        "PASS": 1
    }


def test_metrics_empty_dataset_is_defined(tmp_path):
    database = Database(str(tmp_path / "test.db"))

    calculator = ResearchMetricsCalculator(database)

    metrics = calculator.calculate()

    assert metrics.total_experiments == 0
    assert metrics.unique_targets == 0
    assert metrics.unique_properties == 0
    assert metrics.unique_evaluators == 0

    assert metrics.result_distribution.total == 0
    assert metrics.result_distribution.counts == {}
    assert metrics.result_distribution.proportions == {}

    assert metrics.observation_metrics[
        "average_observations_per_experiment"
    ] == 0.0

    assert metrics.observation_metrics[
        "observation_coverage"
    ] == 0.0
