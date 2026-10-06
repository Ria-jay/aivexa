from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aivexa.storage.database import Database


@dataclass(frozen=True)
class ResearchRecord:
    """
    Stable research representation of one AIVEXA experiment.

    This is derived from the existing experiment/assessment/finding/
    reproduction lineage. It does not replace those domain objects.
    """

    schema_version: str
    experiment_id: str
    target: str

    experiment: dict[str, Any]
    assessment: dict[str, Any] | None
    finding: dict[str, Any] | None
    reproductions: list[dict[str, Any]]

    reproducibility: dict[str, Any]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _fingerprint_payload(
    experiment: dict[str, Any],
    assessment: dict[str, Any] | None,
    finding: dict[str, Any] | None,
    reproductions: list[dict[str, Any]],
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "assessment": assessment,
        "finding": finding,
        "reproductions": reproductions,
        "reproducibility": reproducibility,
    }


def calculate_fingerprint(
    experiment: dict[str, Any],
    assessment: dict[str, Any] | None,
    finding: dict[str, Any] | None,
    reproductions: list[dict[str, Any]],
    reproducibility: dict[str, Any],
) -> str:
    payload = _fingerprint_payload(
        experiment=experiment,
        assessment=assessment,
        finding=finding,
        reproductions=reproductions,
        reproducibility=reproducibility,
    )

    canonical = _canonical_json(payload)

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def build_research_record(
    database: Database,
    experiment_id: str,
) -> ResearchRecord:
    experiment_row = database.connection.execute(
        """
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
            assessment_rationale,
            assessment_evidence,
            reproducibility
        FROM experiments
        WHERE experiment_id = ?
        """,
        (experiment_id,),
    ).fetchone()

    if experiment_row is None:
        raise KeyError(
            f"Experiment '{experiment_id}' was not found."
        )

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
        assessment_rationale,
        assessment_evidence,
        reproducibility,
    ) = experiment_row

    experiment = {
        "experiment_id": experiment_id,
        "target": target,
        "objective": objective,
        "hypothesis": hypothesis,
        "intervention": intervention,
        "context": json.loads(context),
        "created_at": created_at,
        "input_data": input_data,
        "output_data": output_data,
        "result": result,
        "rationale": rationale,
        "confidence": confidence,
        "observations": json.loads(observations),
        "property_name": property_name,
        "property_expectation": property_expectation,
        "comparison_group": comparison_group,
        "evaluator": evaluator,
    }

    assessment = None

    if assessment_id:
        assessment = {
            "assessment_id": assessment_id,
            "result": assessment_result,
            "confidence": assessment_confidence,
            "rationale": assessment_rationale,
            "evidence": (
                json.loads(assessment_evidence)
                if assessment_evidence
                else {}
            ),
        }

    finding_rows = database.connection.execute(
        """
        SELECT
            finding_id,
            assessment_id,
            experiment_id,
            title,
            result,
            confidence,
            affected_properties,
            rationale,
            evidence,
            created_at
        FROM findings
        WHERE experiment_id = ?
        ORDER BY created_at
        """,
        (experiment_id,),
    ).fetchall()

    finding = None

    if finding_rows:
        row = finding_rows[0]

        finding = {
            "finding_id": row[0],
            "assessment_id": row[1],
            "experiment_id": row[2],
            "title": row[3],
            "result": row[4],
            "confidence": row[5],
            "affected_properties": json.loads(row[6]),
            "rationale": row[7],
            "evidence": json.loads(row[8]),
            "created_at": row[9],
        }

    reproduction_rows = database.connection.execute(
        """
        SELECT
            reproduction_id,
            finding_id,
            source_experiment_id,
            follow_up_experiment_id,
            status,
            result,
            confidence,
            rationale,
            evidence,
            created_at
        FROM reproductions
        WHERE source_experiment_id = ?
        ORDER BY created_at
        """,
        (experiment_id,),
    ).fetchall()

    reproductions = [
        {
            "reproduction_id": row[0],
            "finding_id": row[1],
            "source_experiment_id": row[2],
            "follow_up_experiment_id": row[3],
            "status": row[4],
            "result": row[5],
            "confidence": row[6],
            "rationale": row[7],
            "evidence": json.loads(row[8]),
            "created_at": row[9],
        }
        for row in reproduction_rows
    ]

    reproducibility_data = (
        json.loads(reproducibility)
        if reproducibility
        else {}
    )

    reproducibility_data.setdefault(
        "experiment_id",
        experiment_id,
    )
    reproducibility_data.setdefault(
        "input_sha256",
        hashlib.sha256(
            input_data.encode("utf-8")
        ).hexdigest(),
    )

    fingerprint = calculate_fingerprint(
        experiment=experiment,
        assessment=assessment,
        finding=finding,
        reproductions=reproductions,
        reproducibility=reproducibility_data,
    )

    return ResearchRecord(
        schema_version="1.0",
        experiment_id=experiment_id,
        target=target,
        experiment=experiment,
        assessment=assessment,
        finding=finding,
        reproductions=reproductions,
        reproducibility=reproducibility_data,
        fingerprint=fingerprint,
    )


class ResearchDataset:
    """
    Dataset interface over AIVEXA's existing persisted research lineage.
    """

    def __init__(self, database: Database):
        self.database = database

    def record(
        self,
        experiment_id: str,
    ) -> ResearchRecord:
        return build_research_record(
            database=self.database,
            experiment_id=experiment_id,
        )

    def records(self) -> list[ResearchRecord]:
        rows = self.database.connection.execute(
            """
            SELECT experiment_id
            FROM experiments
            ORDER BY created_at, experiment_id
            """
        ).fetchall()

        return [
            self.record(row[0])
            for row in rows
        ]

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        destination = Path(path)

        payload = {
            "schema_version": "1.0",
            "dataset_type": "aivexa-research-dataset",
            "records": [
                record.to_dict()
                for record in self.records()
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
    ) -> Path:
        destination = Path(path)

        lines = [
            json.dumps(
                record.to_dict(),
                sort_keys=True,
                ensure_ascii=False,
            )
            for record in self.records()
        ]

        destination.write_text(
            "\n".join(lines)
            + ("\n" if lines else ""),
            encoding="utf-8",
        )

        return destination
