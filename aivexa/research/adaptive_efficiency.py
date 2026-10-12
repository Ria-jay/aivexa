from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from aivexa.research.experiment_dataset import (
    ExperimentDataset,
    ExperimentDatasetRow,
)
from aivexa.storage.database import Database


@dataclass(frozen=True)
class ExperimentOutcome:
    """One experiment outcome used for efficiency comparison."""

    experiment_id: str
    case_id: str | None
    result: str


@dataclass(frozen=True)
class EfficiencyMetrics:
    """Descriptive efficiency metrics for one evaluation strategy."""

    strategy: str
    experiments: int
    distinct_cases: int
    confirmed_findings: int
    anomalies: int
    inconclusive: int
    execution_failures: int
    findings_per_experiment: float
    case_coverage: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EfficiencyComparison:
    fixed: EfficiencyMetrics
    adaptive: EfficiencyMetrics
    additional_experiments: int
    additional_findings: int
    efficiency_gain: float | None
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed": self.fixed.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "additional_experiments": self.additional_experiments,
            "additional_findings": self.additional_findings,
            "efficiency_gain": self.efficiency_gain,
            "rationale": self.rationale,
        }


def _metrics(
    strategy: str,
    outcomes: list[ExperimentOutcome],
) -> EfficiencyMetrics:
    cases = {
        outcome.case_id
        for outcome in outcomes
        if outcome.case_id
    }

    confirmed = sum(
        1 for o in outcomes if o.result == "FAIL"
    )
    anomalies = sum(
        1 for o in outcomes if o.result == "ANOMALY"
    )
    inconclusive = sum(
        1 for o in outcomes if o.result == "INCONCLUSIVE"
    )
    execution_failures = sum(
        1
        for o in outcomes
        if o.result == "EXECUTION_ERROR"
    )

    experiments = len(outcomes)

    findings_per_experiment = round(
        confirmed / experiments if experiments else 0.0,
        4,
    )

    return EfficiencyMetrics(
        strategy=strategy,
        experiments=experiments,
        distinct_cases=len(cases),
        confirmed_findings=confirmed,
        anomalies=anomalies,
        inconclusive=inconclusive,
        execution_failures=execution_failures,
        findings_per_experiment=findings_per_experiment,
        case_coverage=len(cases),
    )


class AdaptiveEfficiencyAnalyzer:
    """
    Compares an adaptive investigation against an equivalent fixed
    evaluation, using the same outcome representation.

    This is a descriptive measurement. It does not claim adaptive
    evaluation is universally better; it makes the trade-off between
    additional experiments and additional confirmed findings explicit.
    """

    def compare(
        self,
        fixed: list[ExperimentOutcome],
        adaptive: list[ExperimentOutcome],
    ) -> EfficiencyComparison:
        fixed_metrics = _metrics("fixed", fixed)
        adaptive_metrics = _metrics("adaptive", adaptive)

        additional_experiments = (
            adaptive_metrics.experiments
            - fixed_metrics.experiments
        )

        additional_findings = (
            adaptive_metrics.confirmed_findings
            - fixed_metrics.confirmed_findings
        )

        if fixed_metrics.findings_per_experiment > 0:
            adaptive_rate = (
                adaptive_metrics.findings_per_experiment
            )
            fixed_rate = fixed_metrics.findings_per_experiment
            efficiency_gain = round(
                (adaptive_rate - fixed_rate) / fixed_rate,
                4,
            )
        else:
            efficiency_gain = None

        if additional_findings > 0:
            rationale = (
                "Adaptive investigation executed "
                f"{additional_experiments} more experiment(s) and "
                f"produced {additional_findings} more confirmed "
                "finding(s) than the fixed evaluation."
            )
        elif additional_experiments > 0:
            rationale = (
                "Adaptive investigation executed "
                f"{additional_experiments} more experiment(s) without "
                "additional confirmed findings; the extra evidence "
                "refined anomaly/inconclusive classification."
            )
        else:
            rationale = (
                "Adaptive and fixed evaluations produced comparable "
                "experiment counts and findings."
            )

        return EfficiencyComparison(
            fixed=fixed_metrics,
            adaptive=adaptive_metrics,
            additional_experiments=additional_experiments,
            additional_findings=additional_findings,
            efficiency_gain=efficiency_gain,
            rationale=rationale,
        )

    def outcomes_from_rows(
        self,
        rows: list[ExperimentDatasetRow],
    ) -> list[ExperimentOutcome]:
        return [
            ExperimentOutcome(
                experiment_id=row.experiment_id,
                case_id=row.comparison_group
                or row.experiment_id,
                result=row.result,
            )
            for row in rows
        ]

    def compare_from_database(
        self,
        database: Database,
    ) -> EfficiencyComparison:
        """
        Compare fixed and adaptive model-safety experiments persisted
        in the database, using the experiment context marker
        ``adaptive``.
        """
        dataset = ExperimentDataset(database)
        rows = dataset.rows()

        fixed_rows = []
        adaptive_rows = []

        for row in rows:
            context = row.context or {}

            if context.get("workflow") not in {
                "model_safety",
                "adaptive",
            }:
                continue

            if context.get("adaptive") or (
                context.get("adaptive_lineage")
            ):
                adaptive_rows.append(row)
            else:
                fixed_rows.append(row)

        return self.compare(
            fixed=self.outcomes_from_rows(fixed_rows),
            adaptive=self.outcomes_from_rows(adaptive_rows),
        )
