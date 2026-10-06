from aivexa.research.statistics import StatisticalAnalyzer
from aivexa.storage.database import Database


def seed_experiment(
    database: Database,
    *,
    experiment_id: str,
    input_data: str,
    result: str,
    target: str = "model-a",
    property_name: str = "authorization_boundary_preservation",
    evaluator: str = "aivexa-authorization-boundary",
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
            "context": {},
            "created_at": "2026-01-01T00:00:00+00:00",
            "input_data": input_data,
            "output_data": "controlled output",
            "result": result,
            "rationale": "Statistical analysis test.",
            "confidence": "High",
            "observations": [],
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


def test_reproducibility_detects_consistent_repeated_group(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    for index in range(3):
        seed_experiment(
            database,
            experiment_id=f"EXP-{index}",
            input_data="same input",
            result="PASS",
        )

    analyzer = StatisticalAnalyzer(database)

    analysis = analyzer.reproducibility(
        target="model-a",
    )

    assert analysis.total_experiments == 3
    assert analysis.repeated_groups == 1
    assert analysis.consistent_groups == 1
    assert analysis.inconsistent_groups == 0
    assert analysis.consistency_rate == 1.0


def test_reproducibility_detects_inconsistent_results(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
        input_data="same input",
        result="PASS",
    )
    seed_experiment(
        database,
        experiment_id="EXP-002",
        input_data="same input",
        result="FAIL",
    )

    analyzer = StatisticalAnalyzer(database)

    analysis = analyzer.reproducibility(
        target="model-a",
    )

    assert analysis.repeated_groups == 1
    assert analysis.consistent_groups == 0
    assert analysis.inconsistent_groups == 1
    assert analysis.consistency_rate == 0.0

    group = analysis.groups[0]

    assert group.sample_count == 2
    assert group.unique_results == ["FAIL", "PASS"]


def test_different_inputs_are_not_treated_as_repetitions(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
        input_data="input one",
        result="PASS",
    )
    seed_experiment(
        database,
        experiment_id="EXP-002",
        input_data="input two",
        result="PASS",
    )

    analyzer = StatisticalAnalyzer(database)

    analysis = analyzer.reproducibility(
        target="model-a",
    )

    assert analysis.total_experiments == 2
    assert analysis.repeated_groups == 0
    assert analysis.consistency_rate is None


def test_result_summary_reports_rates_and_uncertainty(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))
    analyzer = StatisticalAnalyzer(database)

    summary = analyzer.result_summary(
        [
            "PASS",
            "PASS",
            "PASS",
            "FAIL",
        ]
    )

    assert summary.sample_count == 4
    assert summary.result_counts == {
        "PASS": 3,
        "FAIL": 1,
    }
    assert summary.result_rates["PASS"] == 0.75
    assert summary.result_rates["FAIL"] == 0.25
    assert summary.lower_bound is not None
    assert summary.upper_bound is not None
    assert 0.0 <= summary.lower_bound <= 1.0
    assert 0.0 <= summary.upper_bound <= 1.0


def test_filtered_statistical_summary_uses_existing_dataset_filters(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed_experiment(
        database,
        experiment_id="EXP-001",
        input_data="input one",
        result="PASS",
        property_name="authorization_boundary_preservation",
    )

    seed_experiment(
        database,
        experiment_id="EXP-002",
        input_data="input two",
        result="FAIL",
        property_name="data_boundary_preservation",
        evaluator="aivexa-data-boundary",
    )

    analyzer = StatisticalAnalyzer(database)

    summary = analyzer.result_summary_for(
        target="model-a",
        property_name="authorization_boundary_preservation",
    )

    assert summary.sample_count == 1
    assert summary.result_counts == {"PASS": 1}
    assert summary.result_rates == {"PASS": 1.0}
    assert summary.lower_bound is None
    assert summary.upper_bound is None
