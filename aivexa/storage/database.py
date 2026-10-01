import json
import sqlite3
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: str = "aivexa.db"):
        self.path = Path(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    intervention TEXT NOT NULL,
                    context TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_data TEXT,
                    output_data TEXT,
                    result TEXT,
                    rationale TEXT,
                    confidence TEXT,
                    observations TEXT
                )
                """
            )

    def save_experiment(self, record: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
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
                    observations
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["experiment_id"],
                    record["target"],
                    record["objective"],
                    record["hypothesis"],
                    record["intervention"],
                    json.dumps(record["context"]),
                    record["created_at"],
                    record.get("input_data"),
                    record.get("output_data"),
                    record.get("result"),
                    record.get("rationale"),
                    record.get("confidence"),
                    json.dumps(record.get("observations", [])),
                ),
            )
