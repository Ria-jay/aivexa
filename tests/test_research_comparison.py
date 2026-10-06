from aivexa.research.comparison import CrossTargetComparator
from aivexa.storage.database import Database


def seed(
    database: Database,
    *,
    experiment_id: str,
    target: str,
    result: str = "PASS",
    property_name: str = "authorization_boundary_preservation",
    evaluator: str = "aivexa-authorization-boundary",
    objective: str = "Evaluate authorization preservation",
    intervention: str = "AUTHORIZATION_PROBE",
) -> None:
    database.save_experiment(
        {
            "experiment_id": experiment_id,
            "target": target,
            "objective": objective,
            "hypothesis": (
                "The target preserves the declared authorization boundary."
            ),
            "intervention": intervention,
            "context": {
                "phase": 5,
                "comparison_test": True,
            },
            "created_at": (
                f"2026-01-01T00:00:0{experiment_id[-1]}+00:00"
            ),
            "input_data": "controlled input",
            "output_data": f"response from {target}",
            "result": result,
            "rationale": "Controlled comparison experiment.",
            "confidence": "High",
            "observations": [
                f"Observed {target}."
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


def test_comparison_groups_equivalent_experiments_by_target(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
        result="PASS",
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-b",
        result="FAIL",
    )

    comparator = CrossTargetComparator(database)

    comparisons = comparator.compare()

    assert len(comparisons) == 1

    comparison = comparisons[0]

    assert comparison.comparable is True
    assert len(comparison.groups) == 2

    observed = {
        group.target: group.experiments[0].result
        for group in comparison.groups
    }

    assert observed == {
        "model-a": "PASS",
        "model-b": "FAIL",
    }


def test_comparison_does_not_mix_different_interventions(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
        intervention="AUTHORIZATION_PROBE",
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-b",
        intervention="DATA_BOUNDARY_PROBE",
    )

    comparator = CrossTargetComparator(database)

    comparisons = comparator.compare()

    assert len(comparisons) == 2
    assert all(
        comparison.comparable is False
        for comparison in comparisons
    )


def test_comparison_filters_targets_and_experiments(
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
        target="model-b",
    )

    seed(
        database,
        experiment_id="EXP-003",
        target="model-c",
    )

    comparator = CrossTargetComparator(database)

    comparisons = comparator.compare(
        experiment_ids=[
            "EXP-001",
            "EXP-002",
        ],
        targets=[
            "model-a",
            "model-b",
        ],
    )

    assert len(comparisons) == 1

    targets = {
        group.target
        for group in comparisons[0].groups
    }

    assert targets == {
        "model-a",
        "model-b",
    }


def test_comparison_rejects_mixed_evaluators(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    seed(
        database,
        experiment_id="EXP-001",
        target="model-a",
        evaluator="aivexa-authorization-boundary",
    )

    seed(
        database,
        experiment_id="EXP-002",
        target="model-b",
        evaluator="aivexa-data-boundary",
    )

    comparator = CrossTargetComparator(database)

    comparisons = comparator.compare()

    assert len(comparisons) == 1

    comparison = comparisons[0]

    assert comparison.comparable is False
    assert comparison.comparability_reason == (
        "Experiments use different evaluators."
    )
