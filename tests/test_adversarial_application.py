from aivexa.evaluation.adversarial import (
    AdversarialCaseLibrary,
    TargetCapability,
)
from aivexa.evaluation.adversarial_application import (
    ApplicationAdversarialWorkflow,
)
from aivexa.evaluation.adversarial_profile import (
    AdversarialTargetProfile,
)
from aivexa.evidence.system import SystemInteraction
from aivexa.storage.database import Database
from aivexa.targets.application import AIApplicationTarget


class SecureTarget(AIApplicationTarget):
    def interact(
        self,
        experiment_id,
        user_input,
        context=None,
        authorization_context=None,
    ):
        return SystemInteraction(
            experiment_id=experiment_id,
            user_input=user_input,
            system_output=(
                "I cannot override the authorized "
                "security boundary."
            ),
            context=context or {},
            retrieved_context=[],
            tool_calls=[],
            authorization_context=(
                authorization_context or {}
            ),
            observations=[
                "No unauthorized action observed."
            ],
            metadata={"test_target": "secure"},
        )


class VulnerableDataTarget(AIApplicationTarget):
    def interact(
        self,
        experiment_id,
        user_input,
        context=None,
        authorization_context=None,
    ):
        return SystemInteraction(
            experiment_id=experiment_id,
            user_input=user_input,
            system_output=(
                "Here is the protected account information."
            ),
            context=context or {},
            retrieved_context=[],
            tool_calls=[],
            authorization_context={
                "authorized": False,
                **(authorization_context or {}),
            },
            observations=[
                "Protected data was exposed."
            ],
            metadata={"test_target": "vulnerable"},
        )


def profile(capability: TargetCapability):
    return AdversarialTargetProfile(
        target_name="test-target",
        capabilities={capability},
        authorized=True,
        authorization_note="Authorized test target.",
    )


def test_model_case_uses_application_target(tmp_path):
    database = Database(str(tmp_path / "test.db"))

    workflow = ApplicationAdversarialWorkflow(
        target=SecureTarget(
            name="secure",
            endpoint="test://secure",
        ),
        database=database,
    )

    case = AdversarialCaseLibrary.model_boundary()

    result = workflow.run(
        profile=profile(TargetCapability.MODEL),
        cases=(case,),
    )

    assert result["summary"]["executed"] == 1
    assert result["summary"]["skipped"] == 0


def test_data_boundary_case_uses_real_property_evaluator(
    tmp_path,
):
    database = Database(str(tmp_path / "test.db"))

    workflow = ApplicationAdversarialWorkflow(
        target=VulnerableDataTarget(
            name="vulnerable",
            endpoint="test://vulnerable",
        ),
        database=database,
    )

    case = AdversarialCaseLibrary.data_boundary()

    result = workflow.run(
        profile=AdversarialTargetProfile(
            target_name="test-target",
            capabilities=set(TargetCapability),
            authorized=True,
            authorization_note="Authorized test target.",
        ),
        cases=(case,),
    )

    assert result["summary"]["executed"] == 1

    evaluation = result["results"][0]["evaluation"]

    assert evaluation["property_name"] == (
        "data_boundary_preservation"
    )

    assert "individual_evaluations" in (
        evaluation["evidence"]
    )
