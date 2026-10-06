from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aivexa.research.experiment_dataset import (
    ExperimentDataset,
    ExperimentDatasetRow,
)
from aivexa.storage.database import Database


@dataclass(frozen=True)
class CoverageDimension:
    name: str
    covered: int
    total: int
    proportion: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageMetrics:
    """
    Describes breadth of the observed research dataset.

    Coverage is descriptive. It does not imply that a covered
    property, target, or evaluator was tested comprehensively.
    """

    total_experiments: int

    target_coverage: CoverageDimension
    property_coverage: CoverageDimension
    evaluator_coverage: CoverageDimension
    result_coverage: CoverageDimension
    intervention_coverage: CoverageDimension
    objective_coverage: CoverageDimension

    targets: list[str]
    properties: list[str]
    evaluators: list[str]
    results: list[str]
    interventions: list[str]
    objectives: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_experiments": self.total_experiments,
            "target_coverage": self.target_coverage.to_dict(),
            "property_coverage": self.property_coverage.to_dict(),
            "evaluator_coverage": self.evaluator_coverage.to_dict(),
            "result_coverage": self.result_coverage.to_dict(),
            "intervention_coverage": (
                self.intervention_coverage.to_dict()
            ),
            "objective_coverage": (
                self.objective_coverage.to_dict()
            ),
            "targets": self.targets,
            "properties": self.properties,
            "evaluators": self.evaluators,
            "results": self.results,
            "interventions": self.interventions,
            "objectives": self.objectives,
        }


def _dimension(
    name: str,
    values: set[str],
    total: int,
) -> CoverageDimension:
    covered = len(values)

    proportion = (
        covered / total
        if total
        else 0.0
    )

    return CoverageDimension(
        name=name,
        covered=covered,
        total=total,
        proportion=proportion,
    )


class CoverageCalculator:
    """
    Calculates descriptive coverage over the AIVEXA experiment corpus.

    A dimension's denominator is the number of experiments for the
    dataset-level coverage view. The resulting proportion therefore
    describes the number of distinct observed categories relative
    to the experiment population; it is not a percentage of all
    possible security properties.
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
    ) -> CoverageMetrics:
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
    ) -> CoverageMetrics:
        targets = {
            row.target
            for row in rows
            if row.target
        }

        properties = {
            row.property_name
            for row in rows
            if row.property_name
        }

        evaluators = {
            row.evaluator
            for row in rows
            if row.evaluator
        }

        results = {
            row.result
            for row in rows
            if row.result
        }

        interventions = {
            row.intervention
            for row in rows
            if row.intervention
        }

        objectives = {
            row.objective
            for row in rows
            if row.objective
        }

        total = len(rows)

        return CoverageMetrics(
            total_experiments=total,

            target_coverage=_dimension(
                "targets",
                targets,
                total,
            ),

            property_coverage=_dimension(
                "properties",
                properties,
                total,
            ),

            evaluator_coverage=_dimension(
                "evaluators",
                evaluators,
                total,
            ),

            result_coverage=_dimension(
                "results",
                results,
                total,
            ),

            intervention_coverage=_dimension(
                "interventions",
                interventions,
                total,
            ),

            objective_coverage=_dimension(
                "objectives",
                objectives,
                total,
            ),

            targets=sorted(targets),
            properties=sorted(properties),
            evaluators=sorted(evaluators),
            results=sorted(results),
            interventions=sorted(interventions),
            objectives=sorted(objectives),
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
