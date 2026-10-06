from aivexa.evaluation.authorization import (
    AuthorizationBoundaryEvaluator,
)
from aivexa.evaluation.data_boundary import DataBoundaryEvaluator
from aivexa.experiments.application_runner import (
    ApplicationExperimentRunner,
)
from aivexa.targets.mock_application import MockAIApplicationTarget


class AuthorizedToolApplication(MockAIApplicationTarget):
    def interact(
        self,
        experiment_id,
        user_input,
        context=None,
        authorization_context=None,
    ):
        interaction = super().interact(
            experiment_id=experiment_id,
            user_input=user_input,
            context=context,
            authorization_context=authorization_context,
        )

        from aivexa.evidence.system import SystemInteraction

        return SystemInteraction(
            experiment_id=interaction.experiment_id,
            user_input=interaction.user_input,
            system_output=interaction.system_output,
            context=interaction.context,
            retrieved_context=interaction.retrieved_context,
            tool_calls=[
                {
                    "tool": "search",
                    "arguments": {"query": "test"},
                    "authorized": True,
                }
            ],
            authorization_context=interaction.authorization_context,
            observations=interaction.observations,
            metadata=interaction.metadata,
        )


class AuthorizedDataApplication(MockAIApplicationTarget):
    def interact(
        self,
        experiment_id,
        user_input,
        context=None,
        authorization_context=None,
    ):
        interaction = super().interact(
            experiment_id=experiment_id,
            user_input=user_input,
            context=context,
            authorization_context=authorization_context,
        )

        from aivexa.evidence.system import SystemInteraction

        return SystemInteraction(
            experiment_id=interaction.experiment_id,
            user_input=interaction.user_input,
            system_output=interaction.system_output,
            context=interaction.context,
            retrieved_context=[
                {
                    "source": "user-documents",
                    "scope": "private-user",
                    "content": "Authorized document",
                    "authorized": True,
                }
            ],
            tool_calls=interaction.tool_calls,
            authorization_context=interaction.authorization_context,
            observations=interaction.observations,
            metadata=interaction.metadata,
        )


def test_application_runner_authorization_flow() -> None:
    target = AuthorizedToolApplication()
    evaluator = AuthorizationBoundaryEvaluator()

    runner = ApplicationExperimentRunner(
        target=target,
        evaluator=evaluator,
    )

    interaction, evaluation, evidence = runner.run(
        experiment_id="TEST-APP-AUTH-001",
        user_input="Search for a document.",
        authorization_context={
            "principal": "test-user",
            "role": "user",
            "tools": ["search"],
            "scope": ["public"],
        },
    )

    assert interaction.experiment_id == "TEST-APP-AUTH-001"
    assert evaluation.result.value == "PASS"
    assert evaluation.evaluator == "aivexa-authorization-boundary"
    assert evidence.experiment_id == "TEST-APP-AUTH-001"
    assert evidence.metadata["target_type"] == "ai_application"
    assert "system_interaction" in evidence.metadata
    assert "evaluation" in evidence.metadata


def test_application_runner_data_flow() -> None:
    target = AuthorizedDataApplication()
    evaluator = DataBoundaryEvaluator()

    runner = ApplicationExperimentRunner(
        target=target,
        evaluator=evaluator,
    )

    interaction, evaluation, evidence = runner.run(
        experiment_id="TEST-APP-DATA-001",
        user_input="Find my documents.",
        authorization_context={
            "principal": "test-user",
            "role": "user",
            "data_sources": ["user-documents"],
            "data_scopes": ["private-user"],
        },
    )

    assert interaction.experiment_id == "TEST-APP-DATA-001"
    assert evaluation.result.value == "PASS"
    assert evaluation.evaluator == "aivexa-data-boundary"
    assert len(interaction.retrieved_context) == 1
    assert evidence.metadata["retrieved_context"][0]["source"] == (
        "user-documents"
    )
