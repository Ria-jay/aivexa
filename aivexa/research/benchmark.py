from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aivexa.research.experiment_dataset import ExperimentDataset
from aivexa.storage.database import Database


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    name: str
    domain: str
    objective: str
    hypothesis: str
    intervention: str
    property_name: str
    property_expectation: str
    input_data: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    name: str
    version: str
    description: str
    cases: list[BenchmarkCase]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "case_count": len(self.cases),
            "cases": [
                case.to_dict()
                for case in self.cases
            ],
            "metadata": self.metadata,
        }

    def case(self, case_id: str) -> BenchmarkCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case

        raise KeyError(
            f"Benchmark case '{case_id}' was not found."
        )


@dataclass(frozen=True)
class BenchmarkExecution:
    benchmark_id: str
    benchmark_version: str
    case_id: str
    experiment_id: str
    target: str
    result: str
    confidence: str
    evaluator: str | None
    property_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkBuilder:
    """
    Creates immutable benchmark definitions from explicit cases.

    Benchmark definitions describe evaluation work. They do not
    execute targets or classify vulnerabilities.
    """

    def build(
        self,
        *,
        benchmark_id: str,
        name: str,
        version: str,
        description: str,
        cases: list[BenchmarkCase],
        metadata: dict[str, Any] | None = None,
    ) -> Benchmark:
        if not benchmark_id.strip():
            raise ValueError(
                "benchmark_id must not be empty."
            )

        if not name.strip():
            raise ValueError(
                "name must not be empty."
            )

        if not version.strip():
            raise ValueError(
                "version must not be empty."
            )

        if not cases:
            raise ValueError(
                "A benchmark must contain at least one case."
            )

        case_ids = [
            case.case_id
            for case in cases
        ]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "Benchmark case IDs must be unique."
            )

        return Benchmark(
            benchmark_id=benchmark_id,
            name=name,
            version=version,
            description=description,
            cases=list(cases),
            metadata=dict(metadata or {}),
        )


class BenchmarkExecutionEngine:
    """
    Maps persisted AIVEXA experiments back to benchmark cases.

    Execution itself remains outside this component. This prevents
    benchmark definition from becoming an implicit attack runner.
    """

    def __init__(self, database: Database):
        self.database = database
        self.dataset = ExperimentDataset(database)

    def execute_recorded(
        self,
        *,
        benchmark: Benchmark,
        target: str,
    ) -> list[BenchmarkExecution]:
        rows = self.dataset.rows(
            target=target,
        )

        executions: list[BenchmarkExecution] = []

        for case in benchmark.cases:
            matching = [
                row
                for row in rows
                if row.objective == case.objective
                and row.hypothesis == case.hypothesis
                and row.intervention == case.intervention
                and row.property_name == case.property_name
                and row.input_data == case.input_data
            ]

            for row in matching:
                executions.append(
                    BenchmarkExecution(
                        benchmark_id=benchmark.benchmark_id,
                        benchmark_version=benchmark.version,
                        case_id=case.case_id,
                        experiment_id=row.experiment_id,
                        target=row.target,
                        result=row.result,
                        confidence=row.confidence,
                        evaluator=row.evaluator,
                        property_name=row.property_name,
                    )
                )

        return executions

    def coverage(
        self,
        *,
        benchmark: Benchmark,
        target: str,
    ) -> dict[str, Any]:
        executions = self.execute_recorded(
            benchmark=benchmark,
            target=target,
        )

        executed_case_ids = {
            execution.case_id
            for execution in executions
        }

        total_cases = len(benchmark.cases)
        executed_cases = len(executed_case_ids)

        return {
            "benchmark_id": benchmark.benchmark_id,
            "benchmark_version": benchmark.version,
            "target": target,
            "total_cases": total_cases,
            "executed_cases": executed_cases,
            "unexecuted_cases": [
                case.case_id
                for case in benchmark.cases
                if case.case_id not in executed_case_ids
            ],
            "coverage": (
                executed_cases / total_cases
                if total_cases
                else 0.0
            ),
        }


def load_benchmark(path: str | Path) -> Benchmark:
    source = Path(path)

    payload = json.loads(
        source.read_text(
            encoding="utf-8"
        )
    )

    cases = [
        BenchmarkCase(
            case_id=item["case_id"],
            name=item["name"],
            domain=item["domain"],
            objective=item["objective"],
            hypothesis=item["hypothesis"],
            intervention=item["intervention"],
            property_name=item["property_name"],
            property_expectation=item[
                "property_expectation"
            ],
            input_data=item["input_data"],
            metadata=dict(
                item.get("metadata", {})
            ),
        )
        for item in payload["cases"]
    ]

    return Benchmark(
        benchmark_id=payload["benchmark_id"],
        name=payload["name"],
        version=payload["version"],
        description=payload["description"],
        cases=cases,
        metadata=dict(
            payload.get("metadata", {})
        ),
    )


def save_benchmark(
    benchmark: Benchmark,
    path: str | Path,
) -> Path:
    destination = Path(path)

    destination.write_text(
        json.dumps(
            benchmark.to_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return destination
