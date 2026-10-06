import json
import sqlite3


class Database:
    def __init__(self, path: str = "aivexa.db"):
        self.path = path
        self.connection = sqlite3.connect(self.path)
        self._initialize()

    def _initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS findings (
                finding_id TEXT PRIMARY KEY,
                assessment_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                title TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence TEXT NOT NULL,
                affected_properties TEXT NOT NULL,
                rationale TEXT NOT NULL,
                evidence TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reproductions (
                reproduction_id TEXT PRIMARY KEY,
                finding_id TEXT NOT NULL,
                source_experiment_id TEXT NOT NULL,
                follow_up_experiment_id TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence TEXT NOT NULL,
                rationale TEXT NOT NULL,
                evidence TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                target TEXT NOT NULL,
                objective TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                intervention TEXT NOT NULL,
                context TEXT NOT NULL,
                created_at TEXT NOT NULL,
                input_data TEXT NOT NULL,
                output_data TEXT NOT NULL,
                result TEXT NOT NULL,
                rationale TEXT NOT NULL,
                confidence TEXT NOT NULL,
                observations TEXT NOT NULL,
                property_name TEXT,
                property_expectation TEXT,
                comparison_group TEXT,
                evaluator TEXT,
                assessment_id TEXT,
                assessment_result TEXT,
                assessment_confidence TEXT,
                assessment_rationale TEXT,
                assessment_evidence TEXT,
                reproducibility TEXT
            )
            """
        )

        columns = {
            row[1]
            for row in self.connection.execute(
                "PRAGMA table_info(experiments)"
            )
        }

        migrations = {
            "property_name": (
                "ALTER TABLE experiments "
                "ADD COLUMN property_name TEXT"
            ),
            "property_expectation": (
                "ALTER TABLE experiments "
                "ADD COLUMN property_expectation TEXT"
            ),
            "comparison_group": (
                "ALTER TABLE experiments "
                "ADD COLUMN comparison_group TEXT"
            ),
            "evaluator": (
                "ALTER TABLE experiments "
                "ADD COLUMN evaluator TEXT"
            ),
            "assessment_id": (
                "ALTER TABLE experiments "
                "ADD COLUMN assessment_id TEXT"
            ),
            "assessment_result": (
                "ALTER TABLE experiments "
                "ADD COLUMN assessment_result TEXT"
            ),
            "assessment_confidence": (
                "ALTER TABLE experiments "
                "ADD COLUMN assessment_confidence TEXT"
            ),
            "assessment_rationale": (
                "ALTER TABLE experiments "
                "ADD COLUMN assessment_rationale TEXT"
            ),
            "assessment_evidence": (
                "ALTER TABLE experiments "
                "ADD COLUMN assessment_evidence TEXT"
            ),
            "reproducibility": (
                "ALTER TABLE experiments "
                "ADD COLUMN reproducibility TEXT"
            ),
        }

        for column, statement in migrations.items():
            if column not in columns:
                self.connection.execute(statement)

        self.connection.commit()

    def save_reproduction(self, reproduction) -> None:
        """
        Persist a reproduction record linked to its finding and
        follow-up experiment.
        """
        from datetime import datetime, timezone

        self.connection.execute(
            """
            INSERT OR REPLACE INTO reproductions (
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
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reproduction.reproduction_id,
                reproduction.finding_id,
                reproduction.source_experiment_id,
                reproduction.follow_up_experiment_id,
                reproduction.status.value,
                reproduction.result.value,
                reproduction.confidence,
                reproduction.rationale,
                json.dumps(reproduction.evidence),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        self.connection.commit()

    def save_finding(self, finding) -> None:
        """
        Persist an evidence-backed finding produced from an
        Agent Assessment or another AIVEXA evaluation workflow.
        """
        from datetime import datetime, timezone

        self.connection.execute(
            """
            INSERT OR REPLACE INTO findings (
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
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                finding.finding_id,
                finding.assessment_id,
                finding.experiment_id,
                finding.title,
                finding.result.value,
                finding.confidence,
                json.dumps(
                    finding.affected_properties
                ),
                finding.rationale,
                json.dumps(finding.evidence),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        self.connection.commit()

    def save_assessment(self, assessment) -> None:
        """
        Persist the aggregate Agent Assessment and all individual
        evaluator evidence against its existing experiment.
        """
        serialized_evaluations = []

        for evaluation in assessment.evaluations:
            serialized_evaluations.append(
                {
                    "result": evaluation.result.value,
                    "rationale": evaluation.rationale,
                    "confidence": evaluation.confidence,
                    "property_name": evaluation.property_name,
                    "property_expectation": (
                        evaluation.property_expectation
                    ),
                    "evaluator": evaluation.evaluator,
                    "evidence": evaluation.evidence,
                }
            )

        evidence = dict(assessment.evidence)
        evidence["individual_evaluations"] = (
            serialized_evaluations
        )

        self.connection.execute(
            """
            UPDATE experiments
            SET
                assessment_id = ?,
                assessment_result = ?,
                assessment_confidence = ?,
                assessment_rationale = ?,
                assessment_evidence = ?
            WHERE experiment_id = ?
            """,
            (
                assessment.assessment_id,
                assessment.result.value,
                assessment.confidence,
                assessment.rationale,
                json.dumps(evidence),
                assessment.experiment_id,
            ),
        )

        self.connection.commit()

    def save_evaluation(
        self,
        experiment_id: str,
        assessment_id: str,
        evaluation,
        evidence: dict | None = None,
    ) -> None:
        """
        Persist a generic AIVEXA Evaluation against an experiment.

        The current schema uses assessment_* columns for historical
        compatibility. This allows model-safety and other
        non-Agent evaluation workflows to use the same persistence
        layer without pretending they are AgentAssessment objects.
        """
        merged_evidence = dict(evaluation.evidence)

        if evidence:
            merged_evidence.update(evidence)

        self.connection.execute(
            """
            UPDATE experiments
            SET
                assessment_id = ?,
                assessment_result = ?,
                assessment_confidence = ?,
                assessment_rationale = ?,
                assessment_evidence = ?
            WHERE experiment_id = ?
            """,
            (
                assessment_id,
                evaluation.result.value,
                evaluation.confidence,
                evaluation.rationale,
                json.dumps(merged_evidence),
                experiment_id,
            ),
        )

        self.connection.commit()

    def save_experiment(self, experiment: dict) -> None:
        """
        Persist one experiment.

        The INSERT below deliberately lists exactly 18 columns and
        supplies exactly 18 values. The assessment_* fields are
        updated separately by save_assessment() or save_evaluation().
        """
        columns = (
            "experiment_id",
            "target",
            "objective",
            "hypothesis",
            "intervention",
            "context",
            "created_at",
            "input_data",
            "output_data",
            "result",
            "rationale",
            "confidence",
            "observations",
            "property_name",
            "property_expectation",
            "comparison_group",
            "evaluator",
            "reproducibility",
        )

        values = (
            experiment["experiment_id"],
            experiment["target"],
            experiment["objective"],
            experiment["hypothesis"],
            experiment["intervention"],
            json.dumps(
                experiment.get("context", {})
            ),
            experiment["created_at"],
            experiment["input_data"],
            experiment["output_data"],
            experiment["result"],
            experiment["rationale"],
            experiment["confidence"],
            json.dumps(
                experiment.get("observations", [])
            ),
            experiment.get("property_name"),
            experiment.get("property_expectation"),
            experiment.get("comparison_group"),
            experiment.get("evaluator"),
            json.dumps(
                experiment.get(
                    "reproducibility",
                    {},
                )
            ),
        )

        if len(columns) != len(values):
            raise RuntimeError(
                "Internal database error: experiment "
                f"column/value mismatch "
                f"({len(columns)} columns, "
                f"{len(values)} values)."
            )

        placeholders = ", ".join(
            "?" for _ in values
        )

        self.connection.execute(
            f"""
            INSERT OR REPLACE INTO experiments (
                {", ".join(columns)}
            )
            VALUES ({placeholders})
            """,
            values,
        )

        self.connection.commit()
