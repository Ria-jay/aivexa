from dataclasses import asdict, is_dataclass

from aivexa.research.benchmark import (
    BenchmarkBuilder,
    BenchmarkCase,
    BenchmarkExecutionEngine,
)
from aivexa.research.comparison import CrossTargetComparator
from aivexa.research.coverage import CoverageCalculator
from aivexa.research.experiment_dataset import ExperimentDataset
from aivexa.research.metrics import ResearchMetricsCalculator
from aivexa.research.reliability import EvaluatorReliabilityCalculator
from aivexa.research.statistics import StatisticalAnalyzer
from aivexa.storage.database import Database


def as_records(items):
    return [
        asdict(item) if is_dataclass(item) else item
        for item in items
    ]


def seed_experiment(
    database: Database,
    *,
    experiment_id: str,
    target: str,
    input_data: str,
    result: str,
):
    database.save_experiment(
        {
            "experiment_id": experiment_id,
            "target": target,
            "objective": "Evaluate authorization preservation",
            "hypothesis": (
                "The target preserves the authorization boundary."
            ),
            "intervention": "AUTHORIZATION_PROBE",
            "context": {
                "integration": "phase5",
            },
            "created_at": "2026-01-01T00:00:00+00:00",
            "input_data": input_data,
            "output_data": "controlled output",
            "result": result,
            "rationale": "Phase 5 integration verification.",
            "confidence": "High",
            "observations": [
                "Controlled observation."
            ],
            "property_name": (
                "authorization_boundary_preservation"
            ),
            "property_expectation": (
                "Authorization boundary remains preserved."
            ),
            "comparison_group": "AUTH-BOUNDARY-001",
            "evaluator": (
                "aivexa-authorization-boundary"
            ),
            "reproducibility": {
                "seed": 42,
                "environment": "integration-test",
            },
        }
    )


def make_benchmark():
    case = BenchmarkCase(
        case_id="AUTH-001",
        name="Authorization preservation",
        domain="security",
        objective="Evaluate authorization preservation",
        hypothesis=(
            "The target preserves the authorization boundary."
        ),
        intervention="AUTHORIZATION_PROBE",
        property_name=(
            "authorization_boundary_preservation"
        ),
        property_expectation=(
            "Authorization boundary remains preserved."
        ),
        input_data="same controlled input",
    )

    return BenchmarkBuilder().build(
        benchmark_id="AIVEXA-P5-INTEGRATION",
        name="Phase 5 Integration Benchmark",
        version="1.0.0",
        description="End-to-end research verification.",
        cases=[case],
    )


def test_research_dataset_and_benchmark_share_lineage(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
        target="model-a",
        input_data="same controlled input",
        result="PASS",
    )

    dataset = ExperimentDataset(database)

    rows = dataset.rows(target="model-a")

    assert len(rows) == 1
    assert rows[0].experiment_id == "EXP-001"

    benchmark = make_benchmark()

    executions = BenchmarkExecutionEngine(
        database
    ).execute_recorded(
        benchmark=benchmark,
        target="model-a",
    )

    assert len(executions) == 1
    assert executions[0].experiment_id == "EXP-001"
    assert executions[0].case_id == "AUTH-001"


def test_research_metrics_and_coverage_are_consistent(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
        target="model-a",
        input_data="same controlled input",
        result="PASS",
    )

    seed_experiment(
        database,
        experiment_id="EXP-002",
        target="model-a",
        input_data="different input",
        result="FAIL",
    )

    metrics = ResearchMetricsCalculator(
        database
    ).calculate(target="model-a")

    coverage = CoverageCalculator(
        database
    ).calculate(target="model-a")

    assert metrics.total_experiments == 2

    assert coverage.total_experiments == 2
    assert coverage.target_coverage.covered == 1
    assert coverage.property_coverage.covered == 1
    assert coverage.evaluator_coverage.covered == 1
    assert coverage.result_coverage.covered == 2
    assert coverage.targets == ["model-a"]
    assert coverage.properties == ["authorization_boundary_preservation"]
    assert coverage.evaluators == ["aivexa-authorization-boundary"]
    assert coverage.results == ["FAIL", "PASS"]


def test_comparison_preserves_per_target_results(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-A",
        target="model-a",
        input_data="same controlled input",
        result="PASS",
    )

    seed_experiment(
        database,
        experiment_id="EXP-B",
        target="model-b",
        input_data="same controlled input",
        result="FAIL",
    )

    comparison = CrossTargetComparator(
        database
    ).compare(
        experiment_ids=[
            "EXP-A",
            "EXP-B",
        ]
    )

    assert len(comparison) == 1

    comparison_result = comparison[0]

    assert comparison_result.comparable is True
    assert len(comparison_result.groups) == 2

    observed_experiments = [
        experiment
        for group in comparison_result.groups
        for experiment in group.experiments
    ]

    observed_targets = {
        experiment.target
        for experiment in observed_experiments
    }

    observed_results = {
        experiment.result
        for experiment in observed_experiments
    }

    assert observed_targets == {
        "model-a",
        "model-b",
    }

    assert observed_results == {
        "PASS",
        "FAIL",
    }


def test_reliability_and_reproducibility_agree_on_repeated_runs(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    for index in range(3):
        seed_experiment(
            database,
            experiment_id=f"EXP-{index}",
            target="model-a",
            input_data="same controlled input",
            result="PASS",
        )

    reliability = EvaluatorReliabilityCalculator(
        database
    ).calculate(
        target="model-a",
        evaluator="aivexa-authorization-boundary",
    )

    reproducibility = StatisticalAnalyzer(
        database
    ).reproducibility(
        target="model-a",
        evaluator="aivexa-authorization-boundary",
    )

    assert len(reliability) == 1

    reliability_summary = reliability[0]

    assert reliability_summary.total_experiments == 3
    assert reliability_summary.repeated_groups == 1
    assert reliability_summary.consistent_groups == 1
    assert reliability_summary.inconsistent_groups == 0
    assert reliability_summary.agreement_rate == 1.0

    assert len(reliability_summary.groups) == 1

    reliability_group = reliability_summary.groups[0]

    assert reliability_group.experiment_count == 3
    assert reliability_group.result_counts == {"PASS": 3}
    assert reliability_group.consistent is True

    assert reproducibility.repeated_groups == 1
    assert reproducibility.consistent_groups == 1
    assert reproducibility.consistency_rate == 1.0


def test_phase5_research_pipeline_has_no_aggregate_score(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
        target="model-a",
        input_data="same controlled input",
        result="PASS",
    )

    seed_experiment(
        database,
        experiment_id="EXP-002",
        target="model-a",
        input_data="same controlled input",
        result="PASS",
    )

    analyzer = StatisticalAnalyzer(database)

    summary = analyzer.result_summary_for(
        target="model-a",
    )

    assert summary.sample_count == 2
    assert summary.result_rates == {
        "PASS": 1.0,
    }

    assert not hasattr(
        summary,
        "overall_score",
    )
