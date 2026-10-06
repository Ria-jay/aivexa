from aivexa.evaluation.adversarial import (
    AdversarialCaseLibrary,
    TargetCapability,
)
from aivexa.evaluation.adversarial_profile import (
    AdversarialTargetProfile,
)
from aivexa.evaluation.adversarial_workflow import (
    AdversarialWorkflow,
)
from aivexa.storage.database import Database


class FakeTarget:
    model = "mock-enterprise-ai"

    def generate(self, prompt, options=None):
        return (
            "The request was processed while preserving the "
            "configured authorization boundary."
        )


class FailingTarget:
    model = "mock-failing-ai"

    def generate(self, prompt, options=None):
        raise TimeoutError("mock timeout")


class FakeRunner:
    def __init__(self, target):
        self.target = target

    def run(
        self,
        experiment_id,
        prompt,
        generation_options=None,
    ):
        from aivexa.evidence.models import Evidence

        return Evidence(
            experiment_id=experiment_id,
            input_data=prompt,
            output_data=self.target.generate(
                prompt,
                options=generation_options,
            ),
            observations=[],
            metadata={
                "target_type": "mock",
                "model": self.target.model,
            },
        )


def authorized_profile(*capabilities):
    return AdversarialTargetProfile(
        target_name="mock-enterprise-ai",
        capabilities=frozenset(capabilities),
        authorized=True,
        authorization_note=(
            "Authorized test target for automated evaluation."
        ),
    )


def test_model_case_executes_for_model_target():
    database = Database(":memory:")

    workflow = AdversarialWorkflow(
        runner=FakeRunner(FakeTarget()),
        database=database,
    )

    result = workflow.run(
        profile=authorized_profile(
            TargetCapability.MODEL,
        ),
        cases=(
            AdversarialCaseLibrary.model_boundary(),
        ),
    )

    assert len(result.results) == 1
    assert (
        result.results[0].case.case_id
        == "ADV-MODEL-BOUNDARY-001"
    )


def test_rag_case_requires_rag_capability():
    database = Database(":memory:")

    workflow = AdversarialWorkflow(
        runner=FakeRunner(FakeTarget()),
        database=database,
    )

    result = workflow.run(
        profile=authorized_profile(
            TargetCapability.MODEL,
        ),
        cases=(
            AdversarialCaseLibrary.rag_instruction_boundary(),
        ),
    )

    assert len(result.results) == 0
    assert (
        "ADV-RAG-INSTRUCTION-001"
        in result.skipped_cases
    )


def test_rag_case_executes_for_rag_target():
    database = Database(":memory:")

    workflow = AdversarialWorkflow(
        runner=FakeRunner(FakeTarget()),
        database=database,
    )

    result = workflow.run(
        profile=authorized_profile(
            TargetCapability.RAG,
        ),
        cases=(
            AdversarialCaseLibrary.rag_instruction_boundary(),
        ),
    )

    assert len(result.results) == 1
    assert (
        result.results[0].evaluation.result.value
        == "INCONCLUSIVE"
    )


def test_unauthorized_profile_is_rejected():
    database = Database(":memory:")

    workflow = AdversarialWorkflow(
        runner=FakeRunner(FakeTarget()),
        database=database,
    )

    profile = AdversarialTargetProfile(
        target_name="unauthorized",
        capabilities=frozenset(
            {TargetCapability.MODEL}
        ),
        authorized=False,
    )

    try:
        workflow.run(
            profile=profile,
            cases=(
                AdversarialCaseLibrary.model_boundary(),
            ),
        )
    except PermissionError:
        pass
    else:
        raise AssertionError(
            "Unauthorized adversarial evaluation was accepted."
        )


def test_execution_error_becomes_inconclusive():
    database = Database(":memory:")

    workflow = AdversarialWorkflow(
        runner=FakeRunner(FailingTarget()),
        database=database,
    )

    result = workflow.run(
        profile=authorized_profile(
            TargetCapability.MODEL,
        ),
        cases=(
            AdversarialCaseLibrary.model_boundary(),
        ),
    )

    assert len(result.results) == 1

    evaluation = result.results[0].evaluation

    assert evaluation.result.value == "INCONCLUSIVE"
    assert evaluation.confidence == "Low"

    row = database.connection.execute(
        """
        SELECT result, property_name
        FROM experiments
        WHERE experiment_id = ?
        """,
        (
            result.results[0].experiment_id,
        ),
    ).fetchone()

    assert row is not None
    assert row[0] == "INCONCLUSIVE"
    assert row[1] == "safety_boundary_preservation"
