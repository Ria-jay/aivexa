from __future__ import annotations

from aivexa.evaluation.models import (
    Evaluation,
    ExperimentResult,
)
from aivexa.experiments.runner import ExperimentRunner
from aivexa.research.benchmark import (
    BenchmarkBuilder,
    BenchmarkCase,
)
from aivexa.research.benchmark_runner import BenchmarkRunner
from aivexa.storage.database import Database


class MockTarget:
    name = "benchmark-target"
    model = "benchmark-target"
    endpoint = "mock://benchmark"

    def generate(self, prompt, options=None):
        return "I cannot help with that request."


class FailingTarget(MockTarget):
    def generate(self, prompt, options=None):
        raise RuntimeError("simulated failure")


def _benchmark():
    builder = BenchmarkBuilder()

    cases = [
        BenchmarkCase(
            case_id="CASE-1",
            name="Refusal case",
            domain="safety",
            objective="Preserve refusal.",
            hypothesis="Refusal persists.",
            intervention="direct",
            property_name="safety_boundary_preservation",
            property_expectation="Refuse.",
            input_data="Do the unsafe thing.",
        ),
        BenchmarkCase(
            case_id="CASE-2",
            name="Second case",
            domain="safety",
            objective="Preserve refusal.",
            hypothesis="Refusal persists.",
            intervention="direct",
            property_name="safety_boundary_preservation",
            property_expectation="Refuse.",
            input_data="Do the unsafe thing again.",
        ),
    ]

    return builder.build(
        benchmark_id="BENCH-001",
        name="Test benchmark",
        version="1.0",
        description="Test benchmark.",
        cases=cases,
    )


def _evaluator(case, output):
    return Evaluation(
        result=ExperimentResult.PASS,
        rationale="Recorded by test evaluator.",
        confidence="Medium",
        property_name=case.property_name,
        property_expectation=case.property_expectation,
        evaluator="test-evaluator",
        evidence={"length": len(output)},
    )


def test_benchmark_runner_executes_and_records(tmp_path):
    database = Database(str(tmp_path / "bench.db"))
    benchmark = _benchmark()

    runner = BenchmarkRunner(
        runner=ExperimentRunner(MockTarget()),
        database=database,
        evaluator=_evaluator,
        evaluator_name="test-evaluator",
    )

    run = runner.run(
        benchmark,
        target_config={"provider": "mock"},
        generation_options={"temperature": 0.0},
    )

    assert len(run.executions) == 2
    assert all(
        execution.result == "PASS"
        for execution in run.executions
    )

    coverage = run.coverage(benchmark)
    assert coverage["coverage"] == 1.0
    assert run.result_counts() == {"PASS": 2}

    row = database.connection.execute(
        """
        SELECT context, property_name, result
        FROM experiments
        WHERE experiment_id = ?
        """,
        ("BENCH-001-1.0-CASE-1",),
    ).fetchone()

    assert row is not None
    assert row[1] == "safety_boundary_preservation"
    assert row[2] == "PASS"


def test_benchmark_runner_records_execution_failure(tmp_path):
    database = Database(str(tmp_path / "bench-fail.db"))
    benchmark = _benchmark()

    runner = BenchmarkRunner(
        runner=ExperimentRunner(FailingTarget()),
        database=database,
        evaluator=_evaluator,
    )

    run = runner.run(benchmark)

    assert all(
        execution.execution_status == "EXECUTION_ERROR"
        for execution in run.executions
    )
    assert all(
        execution.result == "INCONCLUSIVE"
        for execution in run.executions
    )


def test_benchmark_runner_without_evaluator_is_inconclusive(
    tmp_path,
):
    database = Database(str(tmp_path / "bench-null.db"))
    benchmark = _benchmark()

    runner = BenchmarkRunner(
        runner=ExperimentRunner(MockTarget()),
        database=database,
    )

    run = runner.run(benchmark)

    assert all(
        execution.result == "INCONCLUSIVE"
        for execution in run.executions
    )
