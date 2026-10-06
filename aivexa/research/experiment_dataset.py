from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aivexa.storage.database import Database


@dataclass(frozen=True)
class ExperimentDatasetRow:
    """
    Stable tabular representation of one persisted AIVEXA experiment.

    This intentionally flattens the experiment record so that the dataset
    can be consumed by analysis tools without requiring knowledge of the
    internal SQLite schema.
    """

    schema_version: str
    experiment_id: str
    target: str
    objective: str
    hypothesis: str
    intervention: str
    created_at: str

    result: str
    rationale: str
    confidence: str

    property_name: str | None
    property_expectation: str | None
    comparison_group: str | None
    evaluator: str | None

    assessment_id: str | None
    assessment_result: str | None
    assessment_confidence: str | None

    input_data: str
    output_data: str

    observation_count: int
    context: dict[str, Any]
    observations: list[str]

    reproducibility: dict[str, Any]


DATASET_FIELDS = [
    "schema_version",
    "experiment_id",
    "target",
    "objective",
    "hypothesis",
    "intervention",
    "created_at",
    "result",
    "rationale",
    "confidence",
    "property_name",
    "property_expectation",
    "comparison_group",
    "evaluator",
    "assessment_id",
    "assessment_result",
    "assessment_confidence",
    "input_data",
    "output_data",
    "observation_count",
    "context",
    "observations",
    "reproducibility",
]


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default

    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _row_from_sql(row: tuple[Any, ...]) -> ExperimentDatasetRow:
    (
        experiment_id,
        target,
        objective,
        hypothesis,
        intervention,
        context,
        created_at,
        input_data,
        output_data,
        result,
        rationale,
        confidence,
        observations,
        property_name,
        property_expectation,
        comparison_group,
        evaluator,
        assessment_id,
        assessment_result,
        assessment_confidence,
        reproducibility,
    ) = row

    context_data = _loads(context, {})
    observations_data = _loads(observations, [])
    reproducibility_data = _loads(reproducibility, {})

    if not isinstance(context_data, dict):
        context_data = {}

    if not isinstance(observations_data, list):
        observations_data = []

    if not isinstance(reproducibility_data, dict):
        reproducibility_data = {}

    return ExperimentDatasetRow(
        schema_version="1.0",
        experiment_id=experiment_id,
        target=target,
        objective=objective,
        hypothesis=hypothesis,
        intervention=intervention,
        created_at=created_at,
        result=result,
        rationale=rationale,
        confidence=confidence,
        property_name=property_name,
        property_expectation=property_expectation,
        comparison_group=comparison_group,
        evaluator=evaluator,
        assessment_id=assessment_id,
        assessment_result=assessment_result,
        assessment_confidence=assessment_confidence,
        input_data=input_data,
        output_data=output_data,
        observation_count=len(observations_data),
        context=context_data,
        observations=observations_data,
        reproducibility=reproducibility_data,
    )


class ExperimentDataset:
    """
    Query and export interface for AIVEXA's persisted experiments.

    This is deliberately separate from ResearchDataset:
    ResearchDataset preserves complete lineage, while this class provides
    a stable experiment-centric analytical dataset.
    """

    def __init__(self, database: Database):
        self.database = database

    def rows(
        self,
        *,
        target: str | None = None,
        result: str | None = None,
        confidence: str | None = None,
        property_name: str | None = None,
        evaluator: str | None = None,
    ) -> list[ExperimentDatasetRow]:
        clauses = []
        parameters: list[str] = []

        if target is not None:
            clauses.append("target = ?")
            parameters.append(target)

        if result is not None:
            clauses.append("result = ?")
            parameters.append(result)

        if confidence is not None:
            clauses.append("confidence = ?")
            parameters.append(confidence)

        if property_name is not None:
            clauses.append("property_name = ?")
            parameters.append(property_name)

        if evaluator is not None:
            clauses.append("evaluator = ?")
            parameters.append(evaluator)

        where_clause = ""

        if clauses:
            where_clause = "WHERE " + " AND ".join(clauses)

        query = f"""
            SELECT
                experiment_id,
                target,
                objective,
                hypothesis,
                intervention,
                context,
                created_at,
                input_data,
                output_data,
                result,
                rationale,
                confidence,
                observations,
                property_name,
                property_expectation,
                comparison_group,
                evaluator,
                assessment_id,
                assessment_result,
                assessment_confidence,
                reproducibility
            FROM experiments
            {where_clause}
            ORDER BY created_at, experiment_id
        """

        rows = self.database.connection.execute(
            query,
            parameters,
        ).fetchall()

        return [
            _row_from_sql(row)
            for row in rows
        ]

    def export_json(
        self,
        path: str | Path,
        **filters: str | None,
    ) -> Path:
        destination = Path(path)
        rows = self.rows(**filters)

        payload = {
            "schema_version": "1.0",
            "dataset_type": "aivexa-experiment-dataset",
            "row_count": len(rows),
            "rows": [
                asdict(row)
                for row in rows
            ],
        }

        destination.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return destination

    def export_jsonl(
        self,
        path: str | Path,
        **filters: str | None,
    ) -> Path:
        destination = Path(path)
        rows = self.rows(**filters)

        lines = [
            json.dumps(
                asdict(row),
                sort_keys=True,
                ensure_ascii=False,
            )
            for row in rows
        ]

        destination.write_text(
            "\n".join(lines)
            + ("\n" if lines else ""),
            encoding="utf-8",
        )

        return destination

    def export_csv(
        self,
        path: str | Path,
        **filters: str | None,
    ) -> Path:
        destination = Path(path)
        rows = self.rows(**filters)

        with destination.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=DATASET_FIELDS,
            )

            writer.writeheader()

            for row in rows:
                data = asdict(row)

                data["context"] = json.dumps(
                    data["context"],
                    sort_keys=True,
                    ensure_ascii=False,
                )

                data["observations"] = json.dumps(
                    data["observations"],
                    ensure_ascii=False,
                )

                data["reproducibility"] = json.dumps(
                    data["reproducibility"],
                    sort_keys=True,
                    ensure_ascii=False,
                )

                writer.writerow(data)

        return destination
