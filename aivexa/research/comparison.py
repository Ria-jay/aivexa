from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aivexa.research.experiment_dataset import (
    ExperimentDataset,
    ExperimentDatasetRow,
)
from aivexa.storage.database import Database


@dataclass(frozen=True)
class ComparisonObservation:
    experiment_id: str
    target: str
    result: str
    confidence: str
    property_name: str | None
    evaluator: str | None
    observation_count: int


@dataclass(frozen=True)
class ComparisonGroup:
    comparison_key: str
    target: str
    experiments: list[ComparisonObservation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_key": self.comparison_key,
            "target": self.target,
            "experiments": [
                asdict(experiment)
                for experiment in self.experiments
            ],
        }


@dataclass(frozen=True)
class CrossTargetComparison:
    comparison_key: str
    property_name: str | None
    evaluator: str | None
    groups: list[ComparisonGroup]
    comparable: bool
    comparability_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_key": self.comparison_key,
            "property_name": self.property_name,
            "evaluator": self.evaluator,
            "comparable": self.comparable,
            "comparability_reason": self.comparability_reason,
            "groups": [
                group.to_dict()
                for group in self.groups
            ],
        }


def _comparison_key(row: ExperimentDatasetRow) -> str:
    """
    Derive a stable comparison key from the declared experiment
    objective and intervention.

    Experiments with different objectives/interventions should not
    silently become part of the same comparative study.
    """
    return (
        f"{row.objective}"
        f"::{row.intervention}"
    )


class CrossTargetComparator:
    """
    Compares equivalent AIVEXA experiments across different targets.

    This component deliberately does not produce rankings or winners.
    It reports whether experiments are structurally comparable and
    preserves each target's observed result independently.
    """

    def __init__(self, database: Database):
        self.database = database
        self.dataset = ExperimentDataset(database)

    def compare(
        self,
        *,
        experiment_ids: list[str] | None = None,
        targets: list[str] | None = None,
        property_name: str | None = None,
        evaluator: str | None = None,
    ) -> list[CrossTargetComparison]:
        rows = self.dataset.rows(
            property_name=property_name,
            evaluator=evaluator,
        )

        if experiment_ids is not None:
            allowed_ids = set(experiment_ids)
            rows = [
                row
                for row in rows
                if row.experiment_id in allowed_ids
            ]

        if targets is not None:
            allowed_targets = set(targets)
            rows = [
                row
                for row in rows
                if row.target in allowed_targets
            ]

        grouped: dict[str, list[ExperimentDatasetRow]] = {}

        for row in rows:
            key = _comparison_key(row)
            grouped.setdefault(key, []).append(row)

        comparisons = []

        for key, group_rows in grouped.items():
            target_names = sorted(
                {
                    row.target
                    for row in group_rows
                }
            )

            groups = []

            for target in target_names:
                target_rows = [
                    row
                    for row in group_rows
                    if row.target == target
                ]

                observations = [
                    ComparisonObservation(
                        experiment_id=row.experiment_id,
                        target=row.target,
                        result=row.result,
                        confidence=row.confidence,
                        property_name=row.property_name,
                        evaluator=row.evaluator,
                        observation_count=row.observation_count,
                    )
                    for row in target_rows
                ]

                groups.append(
                    ComparisonGroup(
                        comparison_key=key,
                        target=target,
                        experiments=observations,
                    )
                )

            property_names = {
                row.property_name
                for row in group_rows
            }

            evaluator_names = {
                row.evaluator
                for row in group_rows
            }

            comparable = (
                len(target_names) >= 2
                and len(property_names) <= 1
                and len(evaluator_names) <= 1
            )

            if len(target_names) < 2:
                reason = (
                    "At least two distinct targets are required."
                )
            elif len(property_names) > 1:
                reason = (
                    "Experiments use different evaluated properties."
                )
            elif len(evaluator_names) > 1:
                reason = (
                    "Experiments use different evaluators."
                )
            else:
                reason = (
                    "Targets share the same objective, intervention, "
                    "property, and evaluator."
                )

            comparisons.append(
                CrossTargetComparison(
                    comparison_key=key,
                    property_name=(
                        next(iter(property_names))
                        if len(property_names) == 1
                        else None
                    ),
                    evaluator=(
                        next(iter(evaluator_names))
                        if len(evaluator_names) == 1
                        else None
                    ),
                    groups=groups,
                    comparable=comparable,
                    comparability_reason=reason,
                )
            )

        return comparisons
