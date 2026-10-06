from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from aivexa.research.experiment_dataset import (
    ExperimentDataset,
    ExperimentDatasetRow,
)
from aivexa.storage.database import Database


@dataclass(frozen=True)
class MetricDistribution:
    total: int
    counts: dict[str, int]
    proportions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchMetrics:
    """
    Descriptive research metrics derived from persisted experiments.

    These metrics describe the observed dataset. They do not rank
    targets, models, evaluators, or security programs.
    """

    total_experiments: int
    unique_targets: int
    unique_properties: int
    unique_evaluators: int

    result_distribution: MetricDistribution
    confidence_distribution: MetricDistribution
    target_distribution: MetricDistribution
    property_distribution: MetricDistribution
    evaluator_distribution: MetricDistribution

    observation_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_experiments": self.total_experiments,
            "unique_targets": self.unique_targets,
            "unique_properties": self.unique_properties,
            "unique_evaluators": self.unique_evaluators,
            "result_distribution": (
                self.result_distribution.to_dict()
            ),
            "confidence_distribution": (
                self.confidence_distribution.to_dict()
            ),
            "target_distribution": (
                self.target_distribution.to_dict()
            ),
            "property_distribution": (
                self.property_distribution.to_dict()
            ),
            "evaluator_distribution": (
                self.evaluator_distribution.to_dict()
            ),
            "observation_metrics": self.observation_metrics,
        }


def _distribution(
    values: list[str],
) -> MetricDistribution:
    counts = Counter(values)
    total = len(values)

    if total == 0:
        proportions = {}
    else:
        proportions = {
            key: count / total
            for key, count in sorted(counts.items())
        }

    return MetricDistribution(
        total=total,
        counts=dict(sorted(counts.items())),
        proportions=proportions,
    )


class ResearchMetricsCalculator:
    """
    Calculates descriptive metrics over AIVEXA experiments.

    Metrics are intentionally decomposed rather than reduced to a
    single score. This keeps research analysis traceable to the
    underlying experiment population.
    """

    def __init__(self, database: Database):
        self.database = database
        self.dataset = ExperimentDataset(database)

    def calculate(
        self,
        *,
        target: str | None = None,
        result: str | None = None,
        confidence: str | None = None,
        property_name: str | None = None,
        evaluator: str | None = None,
    ) -> ResearchMetrics:
        rows = self.dataset.rows(
            target=target,
            result=result,
            confidence=confidence,
            property_name=property_name,
            evaluator=evaluator,
        )

        return self.calculate_rows(rows)

    def calculate_rows(
        self,
        rows: list[ExperimentDatasetRow],
    ) -> ResearchMetrics:
        result_values = [
            row.result
            for row in rows
        ]

        confidence_values = [
            row.confidence
            for row in rows
        ]

        target_values = [
            row.target
            for row in rows
        ]

        property_values = [
            row.property_name
            for row in rows
            if row.property_name is not None
        ]

        evaluator_values = [
            row.evaluator
            for row in rows
            if row.evaluator is not None
        ]

        total_observations = sum(
            row.observation_count
            for row in rows
        )

        experiments_with_observations = sum(
            1
            for row in rows
            if row.observation_count > 0
        )

        experiments_with_assessments = sum(
            1
            for row in rows
            if row.assessment_id is not None
        )

        total = len(rows)

        if total:
            average_observations = (
                total_observations / total
            )
            observation_coverage = (
                experiments_with_observations / total
            )
            assessment_coverage = (
                experiments_with_assessments / total
            )
        else:
            average_observations = 0.0
            observation_coverage = 0.0
            assessment_coverage = 0.0

        return ResearchMetrics(
            total_experiments=total,
            unique_targets=len(set(target_values)),
            unique_properties=len(set(property_values)),
            unique_evaluators=len(set(evaluator_values)),
            result_distribution=_distribution(
                result_values
            ),
            confidence_distribution=_distribution(
                confidence_values
            ),
            target_distribution=_distribution(
                target_values
            ),
            property_distribution=_distribution(
                property_values
            ),
            evaluator_distribution=_distribution(
                evaluator_values
            ),
            observation_metrics={
                "total_observations": float(
                    total_observations
                ),
                "average_observations_per_experiment": (
                    average_observations
                ),
                "experiments_with_observations": float(
                    experiments_with_observations
                ),
                "observation_coverage": (
                    observation_coverage
                ),
                "experiments_with_assessments": float(
                    experiments_with_assessments
                ),
                "assessment_coverage": (
                    assessment_coverage
                ),
            },
        )

    def export_json(
        self,
        path: str,
        **filters: str | None,
    ) -> None:
        import json
        from pathlib import Path

        metrics = self.calculate(**filters)

        Path(path).write_text(
            json.dumps(
                metrics.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
