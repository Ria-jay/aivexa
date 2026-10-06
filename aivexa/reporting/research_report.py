from __future__ import annotations

import json
from typing import Any

from aivexa.storage.database import Database


class ResearchReportGenerator:
    """
    Bug-bounty/researcher-facing report generated from persisted
    AIVEXA evidence.

    It does not submit anything externally.
    """

    def __init__(self, database: Database):
        self.database = database

    def finding_report(
        self,
        finding_id: str,
    ) -> str:
        finding = self.database.connection.execute(
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
            WHERE finding_id = ?
            """,
            (finding_id,),
        ).fetchone()

        if finding is None:
            raise KeyError(
                f"Finding '{finding_id}' was not found."
            )

        (
            finding_id,
            assessment_id,
            experiment_id,
            title,
            result,
            confidence,
            affected_properties,
            rationale,
            evidence,
            created_at,
        ) = finding

        experiment = self.database.connection.execute(
            """
            SELECT
                target,
                objective,
                hypothesis,
                intervention,
                input_data,
                output_data,
                context,
                observations,
                assessment_evidence
            FROM experiments
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

        if experiment is None:
            raise KeyError(
                f"Source experiment '{experiment_id}' was not found."
            )

        (
            target,
            objective,
            hypothesis,
            intervention,
            input_data,
            output_data,
            context,
            observations,
            assessment_evidence,
        ) = experiment

        reproductions = self.database.connection.execute(
            """
            SELECT
                reproduction_id,
                follow_up_experiment_id,
                status,
                result,
                confidence,
                rationale,
                evidence
            FROM reproductions
            WHERE finding_id = ?
            ORDER BY created_at
            """,
            (finding_id,),
        ).fetchall()

        def decode(value: Any, default: Any) -> Any:
            if not value:
                return default
            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return default

        affected = decode(
            affected_properties,
            [],
        )

        finding_evidence = decode(
            evidence,
            {},
        )

        context_data = decode(
            context,
            {},
        )

        observations_data = decode(
            observations,
            [],
        )

        assessment_evidence_data = decode(
            assessment_evidence,
            {},
        )

        lines = [
            "# AIVEXA AI Security Finding Report",
            "",
            f"## Finding: {title}",
            "",
            f"- Finding ID: `{finding_id}`",
            f"- Assessment ID: `{assessment_id}`",
            f"- Source experiment: `{experiment_id}`",
            f"- Target: `{target}`",
            f"- Result: **{result}**",
            f"- Confidence: **{confidence}**",
            f"- Created: `{created_at}`",
            "",
            "## Summary",
            "",
            rationale,
            "",
            "## Affected Security Properties",
            "",
        ]

        if affected:
            lines.extend(
                f"- `{item}`"
                for item in affected
            )
        else:
            lines.append("- None recorded.")

        lines.extend(
            [
                "",
                "## Objective",
                "",
                objective,
                "",
                "## Hypothesis",
                "",
                hypothesis,
                "",
                "## Intervention",
                "",
                intervention,
                "",
                "## Input",
                "",
                "```text",
                str(input_data),
                "```",
                "",
                "## Observed Output",
                "",
                "```text",
                str(output_data),
                "```",
                "",
                "## Observations",
                "",
            ]
        )

        if observations_data:
            lines.extend(
                f"- {item}"
                for item in observations_data
            )
        else:
            lines.append("- None recorded.")

        lines.extend(
            [
                "",
                "## Evidence",
                "",
                "```json",
                json.dumps(
                    finding_evidence,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
                "## Assessment Evidence",
                "",
                "```json",
                json.dumps(
                    assessment_evidence_data,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
                "## Context",
                "",
                "```json",
                json.dumps(
                    context_data,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
                "## Reproduction",
                "",
            ]
        )

        if not reproductions:
            lines.append(
                "No controlled reproduction has been persisted."
            )
        else:
            for reproduction in reproductions:
                (
                    reproduction_id,
                    follow_up_id,
                    status,
                    reproduction_result,
                    reproduction_confidence,
                    reproduction_rationale,
                    reproduction_evidence,
                ) = reproduction

                lines.extend(
                    [
                        f"### `{reproduction_id}`",
                        "",
                        f"- Follow-up experiment: `{follow_up_id}`",
                        f"- Status: **{status}**",
                        f"- Result: **{reproduction_result}**",
                        f"- Confidence: **{reproduction_confidence}**",
                        "",
                        reproduction_rationale,
                        "",
                        "```json",
                        json.dumps(
                            decode(
                                reproduction_evidence,
                                {},
                            ),
                            indent=2,
                            sort_keys=True,
                        ),
                        "```",
                        "",
                    ]
                )

        lines.extend(
            [
                "## Reviewer Notes",
                "",
                "This report describes observed behavior and "
                "preserved evidence from an authorized AIVEXA "
                "assessment. Impact and vulnerability classification "
                "should be independently validated by the program "
                "or system owner.",
                "",
                "---",
                "",
                "Generated by AIVEXA.",
            ]
        )

        return "\n".join(lines)
