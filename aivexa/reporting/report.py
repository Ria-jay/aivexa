import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aivexa.storage.database import Database


class ReportGenerator:
    """
    Generates an evidence-preserving Markdown report directly
    from AIVEXA's persisted experiment lineage.
    """

    def __init__(self, database: Database):
        self.database = database

    def generate(
        self,
        experiment_id: str,
        output_path: str | None = None,
    ) -> str:
        experiment = self.database.connection.execute(
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
                assessment_evidence
            FROM experiments
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

        if experiment is None:
            raise KeyError(
                f"Experiment '{experiment_id}' was not found."
            )

        findings = self.database.connection.execute(
            """
            SELECT
                finding_id,
                assessment_id,
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

        reproductions = self.database.connection.execute(
            """
            SELECT
                reproduction_id,
                finding_id,
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

        (
            eid,
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
        ) = experiment

        context_data = self._json(context, {})
        observations_data = self._json(observations, [])
        assessment_evidence_data = self._json(
            assessment_evidence,
            {},
        )

        lines = [
            "# AIVEXA Security Assessment Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Assessment",
            "",
            f"- Experiment: `{eid}`",
            f"- Target: `{target}`",
            f"- Result: **{result}**",
            f"- Confidence: **{confidence}**",
            f"- Evaluator: `{evaluator}`",
            f"- Created: `{created_at}`",
            "",
            "## Rationale",
            "",
            rationale or "No experiment rationale recorded.",
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
            "## Property",
            "",
            f"- Name: `{property_name}`",
            f"- Expectation: {property_expectation}",
            "",
            "## Input",
            "",
            "```text",
            str(input_data),
            "```",
            "",
            "## Output",
            "",
            "```text",
            str(output_data),
            "```",
            "",
            "## Observations",
            "",
        ]

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
                "## Findings",
                "",
            ]
        )

        if not findings:
            lines.append(
                "No finding was generated from this assessment."
            )
        else:
            for finding in findings:
                (
                    finding_id,
                    finding_assessment_id,
                    title,
                    finding_result,
                    finding_confidence,
                    affected_properties,
                    finding_rationale,
                    finding_evidence,
                    finding_created_at,
                ) = finding

                lines.extend(
                    [
                        f"### {title}",
                        "",
                        f"- Finding ID: `{finding_id}`",
                        f"- Assessment ID: `{finding_assessment_id}`",
                        f"- Result: **{finding_result}**",
                        f"- Confidence: **{finding_confidence}**",
                        f"- Created: `{finding_created_at}`",
                        "",
                        "**Affected properties:**",
                        "",
                    ]
                )

                for prop in self._json(
                    affected_properties,
                    [],
                ):
                    lines.append(f"- `{prop}`")

                lines.extend(
                    [
                        "",
                        "**Rationale:**",
                        "",
                        finding_rationale,
                        "",
                        "**Evidence:**",
                        "",
                        "```json",
                        json.dumps(
                            self._json(
                                finding_evidence,
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
                "## Reproductions",
                "",
            ]
        )

        if not reproductions:
            lines.append(
                "No reproduction record is currently persisted."
            )
        else:
            for reproduction in reproductions:
                (
                    reproduction_id,
                    finding_id,
                    follow_up_id,
                    status,
                    reproduction_result,
                    reproduction_confidence,
                    reproduction_rationale,
                    reproduction_evidence,
                    reproduction_created_at,
                ) = reproduction

                lines.extend(
                    [
                        f"### `{reproduction_id}`",
                        "",
                        f"- Finding: `{finding_id}`",
                        f"- Follow-up experiment: `{follow_up_id}`",
                        f"- Status: **{status}**",
                        f"- Result: **{reproduction_result}**",
                        f"- Confidence: **{reproduction_confidence}**",
                        f"- Created: `{reproduction_created_at}`",
                        "",
                        reproduction_rationale,
                        "",
                        "```json",
                        json.dumps(
                            self._json(
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
                "## Research Lineage",
                "",
                f"- Source experiment: `{eid}`",
                f"- Comparison group: `{comparison_group}`",
                f"- Assessment ID: `{assessment_id}`",
                f"- Assessment result: `{assessment_result}`",
                f"- Assessment confidence: `{assessment_confidence}`",
                f"- Assessment rationale: {assessment_rationale}",
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
                "---",
                "",
                "Generated by AIVEXA.",
            ]
        )

        report = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(report)

        return report

    @staticmethod
    def _json(value: str | None, default: Any) -> Any:
        if not value:
            return default

        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default
