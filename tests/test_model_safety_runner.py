from __future__ import annotations

from aivexa.evaluation.model_safety import (
    ModelSafetySuite,
)
from aivexa.evaluation.model_safety_runner import (
    ModelSafetyRunner,
)
from aivexa.evaluation.models import ExperimentResult
from aivexa.storage.database import Database


class MockModelTarget:
    name = "mock-model"
    model = "mock-model"
    endpoint = "mock://model"

    def generate(
        self,
        prompt,
        options=None,
    ):
        if "hazardous substance" in prompt.lower():
            return (
                "I cannot provide instructions for creating "
                "a hazardous substance."
            )

        return "I cannot provide that information."


def test_model_safety_uses_normal_experiment_workflow(
    tmp_path,
):
    database = Database(
        str(tmp_path / "model-safety.db")
    )

    from aivexa.experiments.runner import ExperimentRunner

    runner = ModelSafetyRunner(
        runner=ExperimentRunner(
            MockModelTarget()
        ),
        database=database,
    )

    case = ModelSafetySuite.refusal_robustness()

    result = runner.run(
        cases=(case,),
        adaptive=False,
    )

    assert result.summary["total_cases"] == 1

    execution = result.executions[0]

    assert execution.evaluation.result == (
        ExperimentResult.PASS
    )

    row = database.connection.execute(
        """
        SELECT
            experiment_id,
            result,
            property_name,
            assessment_id,
            assessment_result,
            assessment_evidence
        FROM experiments
        WHERE experiment_id = ?
        """,
        (
            "SAFETY-REFUSAL-001-EXPERIMENT-001",
        ),
    ).fetchone()

    assert row is not None
    assert row[0] == (
        "SAFETY-REFUSAL-001-EXPERIMENT-001"
    )
    assert row[1] == "PASS"
    assert row[2] == (
        "safety_boundary_preservation"
    )
    assert row[3] == (
        "SAFETY-REFUSAL-001-EXPERIMENT-001-ASSESSMENT"
    )
    assert row[4] == "PASS"
    assert "safety_case" in row[5]


def test_model_safety_finding_is_not_agent_finding(
    tmp_path,
):
    database = Database(
        str(tmp_path / "model-safety-finding.db")
    )

    class UnsafeTarget:
        name = "unsafe-model"
        model = "unsafe-model"
        endpoint = "mock://unsafe"

        def generate(
            self,
            prompt,
            options=None,
        ):
            return (
                "Step 1: combine the materials. "
                "Step 2: heat the mixture. "
                "Step 3: continue the procedure."
            )

    from aivexa.experiments.runner import ExperimentRunner

    runner = ModelSafetyRunner(
        runner=ExperimentRunner(
            UnsafeTarget()
        ),
        database=database,
    )

    case = ModelSafetySuite.refusal_robustness()

    result = runner.run(
        cases=(case,),
        adaptive=False,
    )

    execution = result.executions[0]

    assert execution.evaluation.result == (
        ExperimentResult.FAIL
    )

    assert execution.finding is not None
    assert execution.finding.title == (
        "Model safety boundary violation"
    )

    row = database.connection.execute(
        """
        SELECT title, result
        FROM findings
        WHERE finding_id = ?
        """,
        (
            "SAFETY-REFUSAL-001-EXPERIMENT-001-FINDING",
        ),
    ).fetchone()

    assert row is not None
    assert row[0] == (
        "Model safety boundary violation"
    )
    assert row[1] == "FAIL"


def test_partial_safety_execution_is_preserved(
    tmp_path,
):
    database = Database(
        str(tmp_path / "model-safety-partial.db")
    )

    class PartiallyFailingTarget:
        name = "partial-model"
        model = "partial-model"
        endpoint = "mock://partial"

        calls = 0

        def generate(
            self,
            prompt,
            options=None,
        ):
            self.calls += 1

            if self.calls == 3:
                raise RuntimeError(
                    "simulated execution failure"
                )

            return "I cannot provide that."

    from aivexa.experiments.runner import ExperimentRunner

    target = PartiallyFailingTarget()

    runner = ModelSafetyRunner(
        runner=ExperimentRunner(target),
        database=database,
    )

    case = ModelSafetySuite.refusal_robustness()

    result = runner.run(
        cases=(case,),
        adaptive=False,
    )

    execution = result.executions[0]

    assert execution.case_result.status.value == (
        "PARTIAL_EXECUTION"
    )
