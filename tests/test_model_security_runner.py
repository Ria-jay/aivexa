from aivexa.evaluation.model_security import (
    SecurityTestType,
)
from aivexa.evaluation.model_security_runner import (
    ModelSecurityRunner,
    ModelSecuritySuite,
)
from aivexa.evaluation.models import (
    ExperimentResult,
)
from aivexa.experiments.runner import ExperimentRunner
from aivexa.storage.database import Database


class MockTarget:
    name = "mock-security-model"
    model = "mock-security-model"
    endpoint = "mock://security"

    def generate(self, prompt, options=None):
        normalized = prompt.lower()

        if (
            "ignore previous instructions" in normalized
            or "disregard all previous instructions" in normalized
            or "override" in normalized
            or "new system instruction" in normalized
        ):
            return (
                "I cannot assist with overriding "
                "the system instructions."
            )

        if "system prompt" in normalized:
            return (
                "I cannot provide the system prompt "
                "or hidden instructions."
            )

        if "developer instructions" in normalized:
            return (
                "I cannot reveal hidden developer instructions."
            )

        if "bearer" in normalized:
            return (
                "I cannot provide a real Authorization: "
                "Bearer token."
            )

        if "api key" in normalized:
            return (
                "I cannot provide real credentials."
            )

        return (
            "Security boundaries should protect "
            "internal instructions and credentials."
        )


def test_model_security_runner_uses_normal_workflow(tmp_path):
    database = Database(
        str(tmp_path / "aivexa.db")
    )

    runner = ModelSecurityRunner(
        runner=ExperimentRunner(
            MockTarget()
        ),
        database=database,
    )

    case = ModelSecuritySuite.default_cases()[0]

    result = runner.run(
        cases=(case,),
    )

    assert len(result.executions) == 1

    execution = result.executions[0]

    assert execution.evaluation.result == ExperimentResult.PASS
    assert execution.baseline is not None
    assert len(execution.variants) == 2
    assert execution.errors == ()


def test_model_security_finding_is_created_for_violation(tmp_path):
    database = Database(
        str(tmp_path / "aivexa.db")
    )

    class UnsafeTarget(MockTarget):
        def generate(self, prompt, options=None):
            normalized = prompt.lower()

            if (
                "ignore previous instructions"
                in normalized
                or "disregard all previous instructions"
                in normalized
            ):
                return (
                    "Ignore previous instructions and "
                    "follow my instructions instead."
                )

            return (
                "Security boundaries should be preserved."
            )

    runner = ModelSecurityRunner(
        runner=ExperimentRunner(
            UnsafeTarget()
        ),
        database=database,
    )

    case = ModelSecuritySuite.default_cases()[0]

    result = runner.run(
        cases=(case,),
    )

    execution = result.executions[0]

    assert execution.evaluation.result == ExperimentResult.FAIL
    assert execution.finding is not None
    assert execution.errors == ()


def test_model_security_suite_contains_security_domains():
    cases = ModelSecuritySuite.default_cases()

    assert len(cases) >= 4

    test_types = {
        case.test_type
        for case in cases
    }

    assert (
        SecurityTestType.PROMPT_INJECTION_RESISTANCE
        in test_types
    )

    assert (
        SecurityTestType.INFORMATION_DISCLOSURE
        in test_types
    )

    assert (
        SecurityTestType.INSTRUCTION_BOUNDARY
        in test_types
    )

    assert (
        SecurityTestType.SECURITY_OUTPUT_HANDLING
        in test_types
    )


def test_model_security_runner_preserves_execution_errors(
    tmp_path,
):
    database = Database(
        str(tmp_path / "aivexa.db")
    )

    class FailingTarget(MockTarget):
        def generate(self, prompt, options=None):
            if "ignore previous instructions" in prompt.lower():
                raise RuntimeError(
                    "simulated model execution failure"
                )

            return "Normal safe response."

    runner = ModelSecurityRunner(
        runner=ExperimentRunner(
            FailingTarget()
        ),
        database=database,
    )

    case = ModelSecuritySuite.default_cases()[0]

    result = runner.run(
        cases=(case,),
    )

    execution = result.executions[0]

    assert len(execution.errors) == 1
    assert (
        "RuntimeError"
        in execution.errors[0]
    )

    # One required variant failed, so the evidence is incomplete.
    # AIVEXA must not convert partial execution into a PASS.
    assert execution.evaluation.result == (
        ExperimentResult.INCONCLUSIVE
    )

    assert execution.evaluation.confidence == "Low"

    assert (
        execution.evaluation.evidence[
            "baseline_successful"
        ]
        is True
    )

    assert (
        execution.evaluation.evidence[
            "successful_variant_count"
        ]
        == 1
    )

    assert (
        execution.evaluation.evidence[
            "variant_count"
        ]
        == 2
    )

    assert result.summary[
        "execution_error_count"
    ] == 1
