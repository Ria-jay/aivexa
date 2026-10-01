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
                evaluator TEXT
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
            "property_name": "ALTER TABLE experiments ADD COLUMN property_name TEXT",
            "property_expectation": (
                "ALTER TABLE experiments ADD COLUMN property_expectation TEXT"
            ),
            "comparison_group": (
                "ALTER TABLE experiments ADD COLUMN comparison_group TEXT"
            ),
            "evaluator": (
                "ALTER TABLE experiments ADD COLUMN evaluator TEXT"
            ),
        }

        for column, statement in migrations.items():
            if column not in columns:
                self.connection.execute(statement)

        self.connection.commit()

    def save_experiment(self, experiment: dict) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO experiments (
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
                evaluator
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment["experiment_id"],
                experiment["target"],
                experiment["objective"],
                experiment["hypothesis"],
                experiment["intervention"],
                json.dumps(experiment.get("context", {})),
                experiment["created_at"],
                experiment["input_data"],
                experiment["output_data"],
                experiment["result"],
                experiment["rationale"],
                experiment["confidence"],
                json.dumps(experiment.get("observations", [])),
                experiment.get("property_name"),
                experiment.get("property_expectation"),
                experiment.get("comparison_group"),
                experiment.get("evaluator"),
            ),
        )

        self.connection.commit()
