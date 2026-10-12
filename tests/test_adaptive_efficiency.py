from __future__ import annotations

from aivexa.research.adaptive_efficiency import (
    AdaptiveEfficiencyAnalyzer,
    ExperimentOutcome,
)


def test_compare_counts_additional_findings():
    fixed = [
        ExperimentOutcome("f1", "case-1", "PASS"),
        ExperimentOutcome("f2", "case-2", "PASS"),
    ]

    adaptive = [
        ExperimentOutcome("a1", "case-1", "FAIL"),
        ExperimentOutcome("a2", "case-2", "ANOMALY"),
        ExperimentOutcome("a3", "case-3", "INCONCLUSIVE"),
    ]

    comparison = AdaptiveEfficiencyAnalyzer().compare(
        fixed=fixed,
        adaptive=adaptive,
    )

    assert comparison.fixed.experiments == 2
    assert comparison.adaptive.experiments == 3
    assert comparison.additional_experiments == 1
    assert comparison.additional_findings == 1
    assert "more confirmed finding" in comparison.rationale


def test_compare_without_findings_is_descriptive():
    comparison = AdaptiveEfficiencyAnalyzer().compare(
        fixed=[ExperimentOutcome("f1", "c1", "PASS")],
        adaptive=[],
    )

    assert comparison.additional_experiments == -1
    assert comparison.efficiency_gain is None
    assert comparison.adaptive.experiments == 0


def test_compare_from_database(tmp_path):
    from aivexa.evaluation.model_safety import ModelSafetySuite
    from aivexa.evaluation.model_safety_runner import (
        ModelSafetyRunner,
    )
    from aivexa.experiments.runner import ExperimentRunner
    from aivexa.storage.database import Database

    class UnsafeTarget:
        name = "eff-target"
        model = "eff-target"
        endpoint = "mock://eff"

        def generate(self, prompt, options=None):
            return (
                "Step 1: combine the materials. "
                "Step 2: heat the mixture."
            )

    database = Database(str(tmp_path / "eff.db"))

    ModelSafetyRunner(
        runner=ExperimentRunner(UnsafeTarget()),
        database=database,
        max_adaptive_cycles=2,
    ).run(
        cases=(ModelSafetySuite.refusal_robustness(),),
        adaptive=True,
    )

    comparison = AdaptiveEfficiencyAnalyzer().compare_from_database(
        database
    )

    # The adaptive run persisted its own follow-up experiment.
    assert comparison.adaptive.experiments >= 1
    assert comparison.fixed.experiments >= 1
