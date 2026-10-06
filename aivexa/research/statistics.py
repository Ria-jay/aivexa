from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from aivexa.research.experiment_dataset import ExperimentDataset
from aivexa.storage.database import Database


@dataclass(frozen=True)
class ReproducibilityGroup:
    group_key: str
    experiment_ids: list[str]
    results: list[str]
    unique_results: list[str]
    sample_count: int
    consistent: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StatisticalSummary:
    sample_count: int
    result_counts: dict[str, int]
    result_rates: dict[str, float]
    confidence: str
    lower_bound: float | None
    upper_bound: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReproducibilityAnalysis:
    total_experiments: int
    repeated_groups: int
    consistent_groups: int
    inconsistent_groups: int
    consistency_rate: float | None
    groups: list[ReproducibilityGroup]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_experiments": self.total_experiments,
            "repeated_groups": self.repeated_groups,
            "consistent_groups": self.consistent_groups,
            "inconsistent_groups": self.inconsistent_groups,
            "consistency_rate": self.consistency_rate,
            "groups": [
                group.to_dict()
                for group in self.groups
            ],
        }


class StatisticalAnalyzer:
    """
    Small-sample descriptive statistics for AIVEXA research data.

    These measurements describe the observed dataset. They do not
    establish universal model properties or statistical significance.
    """

    def __init__(self, database: Database):
        self.database = database
        self.dataset = ExperimentDataset(database)

    @staticmethod
    def _group_key(row: Any) -> str:
        return "|".join(
            [
                str(row.target),
                str(row.objective),
                str(row.hypothesis),
                str(row.intervention),
                str(row.property_name),
                str(row.input_data),
                str(row.evaluator),
            ]
        )

    def reproducibility(
        self,
        *,
        target: str | None = None,
        property_name: str | None = None,
        evaluator: str | None = None,
        min_repetitions: int = 2,
    ) -> ReproducibilityAnalysis:
        rows = self.dataset.rows(
            target=target,
            property_name=property_name,
            evaluator=evaluator,
        )

        grouped: dict[str, list[Any]] = {}

        for row in rows:
            key = self._group_key(row)
            grouped.setdefault(key, []).append(row)

        groups: list[ReproducibilityGroup] = []

        for key, repeated_rows in grouped.items():
            if len(repeated_rows) < min_repetitions:
                continue

            results = [
                str(row.result)
                for row in repeated_rows
            ]

            groups.append(
                ReproducibilityGroup(
                    group_key=key,
                    experiment_ids=[
                        row.experiment_id
                        for row in repeated_rows
                    ],
                    results=results,
                    unique_results=sorted(set(results)),
                    sample_count=len(results),
                    consistent=len(set(results)) == 1,
                )
            )

        consistent = sum(
            group.consistent
            for group in groups
        )

        repeated_count = len(groups)

        return ReproducibilityAnalysis(
            total_experiments=len(rows),
            repeated_groups=repeated_count,
            consistent_groups=consistent,
            inconsistent_groups=(
                repeated_count - consistent
            ),
            consistency_rate=(
                consistent / repeated_count
                if repeated_count
                else None
            ),
            groups=groups,
        )

    @staticmethod
    def result_summary(
        results: list[str],
        *,
        confidence: str = "descriptive",
    ) -> StatisticalSummary:
        counts: dict[str, int] = {}

        for result in results:
            counts[result] = counts.get(result, 0) + 1

        sample_count = len(results)

        rates = {
            result: count / sample_count
            for result, count in counts.items()
        } if sample_count else {}

        lower_bound: float | None = None
        upper_bound: float | None = None

        if sample_count:
            observed = max(counts.values()) / sample_count

            if sample_count >= 2:
                z = 1.96
                denominator = 1 + (z * z / sample_count)
                centre = (
                    observed
                    + (z * z / (2 * sample_count))
                ) / denominator

                margin = (
                    z
                    * math.sqrt(
                        (
                            observed
                            * (1 - observed)
                            / sample_count
                        )
                        + (
                            z * z
                            / (4 * sample_count * sample_count)
                        )
                    )
                    / denominator
                )

                lower_bound = max(0.0, centre - margin)
                upper_bound = min(1.0, centre + margin)

        return StatisticalSummary(
            sample_count=sample_count,
            result_counts=counts,
            result_rates=rates,
            confidence=confidence,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def result_summary_for(
        self,
        *,
        target: str | None = None,
        property_name: str | None = None,
        evaluator: str | None = None,
    ) -> StatisticalSummary:
        rows = self.dataset.rows(
            target=target,
            property_name=property_name,
            evaluator=evaluator,
        )

        return self.result_summary(
            [
                str(row.result)
                for row in rows
            ]
        )
