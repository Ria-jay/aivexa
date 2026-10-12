from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.experiments.runner import ExperimentRunner
from aivexa.research.benchmark import Benchmark, BenchmarkCase
from aivexa.storage.database import Database


BenchmarkEvaluator = Callable[
    [BenchmarkCase, str],
    Evaluation,
]


@dataclass(frozen=True)
class BenchmarkRunExecution:
    case_id: str
    experiment_id: str
    target: str
    output: str | None
    result: str
    confidence: str
    rationale: str
    execution_status: str
    evaluator: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkRun:
    benchmark_id: str
    benchmark_version: str
    target: str
    target_config: dict[str, Any]
    generation_options: dict[str, Any]
    executions: list[BenchmarkRunExecution] = field(
        default_factory=list
    )
    executed_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def coverage(self, benchmark: Benchmark) -> dict[str, Any]:
        executed_ids = {
            execution.case_id
            for execution in self.executions
        }

        total = len(benchmark.cases)

        return {
            "total_cases": total,
            "executed_cases": len(executed_ids),
            "unexecuted_cases": [
                case.case_id
                for case in benchmark.cases
                if case.case_id not in executed_ids
            ],
            "coverage": (
                len(executed_ids) / total if total else 0.0
            ),
        }

    def result_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for execution in self.executions:
            counts[execution.result] = (
                counts.get(execution.result, 0) + 1
            )

        return counts

    def to_dict(self, benchmark: Benchmark) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "target": self.target,
            "target_config": self.target_config,
            "generation_options": self.generation_options,
            "executed_at": self.executed_at,
            "executions": [
                execution.to_dict()
                for execution in self.executions
            ],
            "coverage": self.coverage(benchmark),
            "result_counts": self.result_counts(),
        }


class BenchmarkRunner:
    """
    Executes a versioned benchmark against a model target and records
    each execution with its configuration and provenance.

    Evaluation is delegated to a caller-supplied evaluator so the
    benchmark layer does not pretend to have ground truth it does not
    have. When no evaluator is supplied, each case is recorded as
    INCONCLUSIVE with an explicit note rather than an assumed PASS.
    """

    def __init__(
        self,
        runner: ExperimentRunner,
        database: Database,
        evaluator: BenchmarkEvaluator | None = None,
        evaluator_name: str = "aivexa-benchmark",
    ):
        self.runner = runner
        self.database = database
        self.evaluator = evaluator
        self.evaluator_name = evaluator_name

    def run(
        self,
        benchmark: Benchmark,
        target_config: dict[str, Any] | None = None,
        generation_options: dict[str, Any] | None = None,
    ) -> BenchmarkRun:
        executions: list[BenchmarkRunExecution] = []

        for case in benchmark.cases:
            execution = self._run_case(
                benchmark=benchmark,
                case=case,
                target_config=target_config or {},
                generation_options=generation_options,
            )
            executions.append(execution)

        return BenchmarkRun(
            benchmark_id=benchmark.benchmark_id,
            benchmark_version=benchmark.version,
            target=self.runner.target.name,
            target_config=dict(target_config or {}),
            generation_options=dict(generation_options or {}),
            executions=executions,
        )

    def _run_case(
        self,
        benchmark: Benchmark,
        case: BenchmarkCase,
        target_config: dict[str, Any],
        generation_options: dict[str, Any] | None,
    ) -> BenchmarkRunExecution:
        experiment_id = (
            f"{benchmark.benchmark_id}"
            f"-{benchmark.version}"
            f"-{case.case_id}"
        )

        output: str | None = None
        execution_status = "COMPLETED"
        errors: list[str] = []

        try:
            evidence = self.runner.run(
                experiment_id=experiment_id,
                prompt=case.input_data,
                generation_options=generation_options,
            )
            output = evidence.output_data
        except Exception as exc:
            execution_status = "EXECUTION_ERROR"
            errors.append(
                f"{type(exc).__name__}: {exc}"
            )

        if output is None:
            evaluation = Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "The benchmark case could not be executed, so no "
                    "result can be established."
                ),
                confidence="Low",
                property_name=case.property_name,
                property_expectation=(
                    case.property_expectation
                ),
                evaluator=self.evaluator_name,
                evidence={"errors": errors},
            )
        elif self.evaluator is not None:
            evaluation = self.evaluator(case, output)
        else:
            evaluation = Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "No benchmark evaluator was configured for this "
                    "run; the response was recorded without a "
                    "classified result."
                ),
                confidence="Low",
                property_name=case.property_name,
                property_expectation=(
                    case.property_expectation
                ),
                evaluator=self.evaluator_name,
                evidence={
                    "response_length": len(output),
                },
            )

        self.database.save_experiment(
            {
                "experiment_id": experiment_id,
                "target": self.runner.target.name,
                "objective": case.objective,
                "hypothesis": case.hypothesis,
                "intervention": case.intervention,
                "context": {
                    "workflow": "benchmark",
                    "benchmark_id": benchmark.benchmark_id,
                    "benchmark_version": benchmark.version,
                    "case_domain": case.domain,
                    "target_config": target_config,
                    "case_metadata": case.metadata,
                },
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "input_data": case.input_data,
                "output_data": output or "",
                "result": evaluation.result.value,
                "rationale": evaluation.rationale,
                "confidence": evaluation.confidence,
                "observations": list(errors),
                "property_name": case.property_name,
                "property_expectation": (
                    case.property_expectation
                ),
                "comparison_group": case.case_id,
                "evaluator": evaluation.evaluator,
                "reproducibility": {
                    "benchmark_id": benchmark.benchmark_id,
                    "benchmark_version": benchmark.version,
                    "generation_options": (
                        generation_options or {}
                    ),
                },
            }
        )

        self.database.save_evaluation(
            experiment_id=experiment_id,
            assessment_id=f"{experiment_id}-ASSESSMENT",
            evaluation=evaluation,
            evidence={
                "benchmark": {
                    "benchmark_id": benchmark.benchmark_id,
                    "benchmark_version": benchmark.version,
                    "case_id": case.case_id,
                },
                "execution_status": execution_status,
            },
        )

        return BenchmarkRunExecution(
            case_id=case.case_id,
            experiment_id=experiment_id,
            target=self.runner.target.name,
            output=output,
            result=evaluation.result.value,
            confidence=evaluation.confidence,
            rationale=evaluation.rationale,
            execution_status=execution_status,
            evaluator=evaluation.evaluator,
        )
