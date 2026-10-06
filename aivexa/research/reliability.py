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
class ReliabilityGroup:
    group_key: str
    target: str
    property_name: str | None
    evaluator: str | None
    experiment_count: int
    result_counts: dict[str, int]
    agreement: float
    consistent: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluatorReliability:
    evaluator: str
    total_experiments: int
    repeated_groups: int
    consistent_groups: int
    inconsistent_groups: int
    agreement_rate: float
    groups: list[ReliabilityGroup]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "total_experiments": self.total_experiments,
            "repeated_groups": self.repeated_groups,
            "consistent_groups": self.consistent_groups,
            "inconsistent_groups": self.inconsistent_groups,
            "agreement_rate": self.agreement_rate,
            "groups": [
                group.to_dict()
                for group in self.groups
            ],
        }


def _group_key(row: ExperimentDatasetRow) -> str:
    """
    Identify repeated evaluations of the same declared research setup.

    The key deliberately excludes experiment_id and created_at so
    separate runs of the same setup can be compared.
    """
    return "::".join(
        [
            row.target,
            row.objective,
            row.hypothesis,
            row.intervention,
            row.property_name or "",
            row.evaluator or "",
            row.input_data,
        ]
    )


def _agreement(results: list[str]) -> float:
    if not results:
        return 0.0

    counts = Counter(results)
    return max(counts.values()) / len(results)


class EvaluatorReliabilityCalculator:
    """
    Measures consistency of evaluator results across repeated,
    structurally equivalent experiments.

    This is a consistency measure, not a validity or correctness
    measure.
    """

    def __init__(self, database: Database):
        self.database = database
        self.dataset = ExperimentDataset(database)

    def calculate(
        self,
        *,
        evaluator: str | None = None,
        target: str | None = None,
        property_name: str | None = None,
    ) -> list[EvaluatorReliability]:
        rows = self.dataset.rows(
            evaluator=evaluator,
            target=target,
            property_name=property_name,
        )

        grouped: dict[str, list[ExperimentDatasetRow]] = {}

        for row in rows:
            key = _group_key(row)
            grouped.setdefault(key, []).append(row)

        evaluator_groups: dict[
            str,
            list[ReliabilityGroup],
        ] = {}

        for key, group_rows in grouped.items():
            if len(group_rows) < 2:
                continue

            evaluator_name = group_rows[0].evaluator

            if evaluator_name is None:
                continue

            result_counts = Counter(
                row.result
                for row in group_rows
            )

            results = [
                row.result
                for row in group_rows
            ]

            agreement = _agreement(results)

            reliability_group = ReliabilityGroup(
                group_key=key,
                target=group_rows[0].target,
                property_name=group_rows[0].property_name,
                evaluator=evaluator_name,
                experiment_count=len(group_rows),
                result_counts=dict(
                    sorted(result_counts.items())
                ),
                agreement=agreement,
                consistent=len(result_counts) == 1,
            )

            evaluator_groups.setdefault(
                evaluator_name,
                [],
            ).append(reliability_group)

        output = []

        all_rows = rows

        evaluators = sorted(
            {
                row.evaluator
                for row in all_rows
                if row.evaluator is not None
            }
        )

        for evaluator_name in evaluators:
            evaluator_rows = [
                row
                for row in all_rows
                if row.evaluator == evaluator_name
            ]

            groups = sorted(
                evaluator_groups.get(
                    evaluator_name,
                    [],
                ),
                key=lambda group: group.group_key,
            )

            repeated_groups = len(groups)

            consistent_groups = sum(
                1
                for group in groups
                if group.consistent
            )

            inconsistent_groups = (
                repeated_groups
                - consistent_groups
            )

            agreement_rate = (
                consistent_groups / repeated_groups
                if repeated_groups
                else 0.0
            )

            output.append(
                EvaluatorReliability(
                    evaluator=evaluator_name,
                    total_experiments=len(
                        evaluator_rows
                    ),
                    repeated_groups=repeated_groups,
                    consistent_groups=consistent_groups,
                    inconsistent_groups=inconsistent_groups,
                    agreement_rate=agreement_rate,
                    groups=groups,
                )
            )

        return output

    def export_json(
        self,
        path: str,
        **filters: str | None,
    ) -> None:
        import json
        from pathlib import Path

        reliability = self.calculate(**filters)

        Path(path).write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "dataset_type": (
                        "aivexa-evaluator-reliability"
                    ),
                    "evaluators": [
                        item.to_dict()
                        for item in reliability
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
