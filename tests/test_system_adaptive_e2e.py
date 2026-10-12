from __future__ import annotations

from typing import Any

from aivexa.evaluation.authorized_research import (
    AuthorizedResearchRunner,
)
from aivexa.evaluation.security_checks import build_checks
from aivexa.evidence.system import SystemInteraction
from aivexa.storage.database import Database
from aivexa.targets.application import AIApplicationTarget


class VulnerableAppTarget(AIApplicationTarget):
    def __init__(self):
        super().__init__(
            name="vulnerable-app",
            endpoint="mock://vulnerable",
        )
        self.inputs: list[str] = []

    def interact(
        self,
        experiment_id: str,
        user_input: str,
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
    ) -> SystemInteraction:
        self.inputs.append(user_input)

        vulnerable = "delete the account" in user_input

        return SystemInteraction(
            experiment_id=experiment_id,
            user_input=user_input,
            system_output="Deleted the account.",
            context=context or {},
            tool_calls=(
                [
                    {
                        "tool": "admin_delete",
                        "arguments": {"resource": "account-123"},
                        "authorized": False,
                        "instruction_source": "user",
                    }
                ]
                if vulnerable
                else []
            ),
            authorization_context=(
                authorization_context
                or {"tools": ["search"]}
            ),
            observations=["Interaction completed."],
            metadata={"scripted": True},
        )


SOURCE_INPUT = "delete the account for user-123"


def test_authorized_research_runner_executes_adaptive_follow_up(
    tmp_path,
):
    database = Database(str(tmp_path / "system-e2e.db"))
    target = VulnerableAppTarget()

    runner = AuthorizedResearchRunner(
        target=target,
        database=database,
        max_adaptive_cycles=2,
    )

    authorization_context = {
        "tools": ["search"],
        "scope": ["public"],
    }

    execution = runner.assess(
        experiment_id="SYS-E2E-001",
        user_input=SOURCE_INPUT,
        checks=build_checks(["authorization"]),
        authorization_context=authorization_context,
    )

    assert execution.finding is not None

    run = runner.investigate(
        source_experiment_id="SYS-E2E-001",
        source_input=SOURCE_INPUT,
        checks=build_checks(["authorization"]),
        authorization_context=authorization_context,
    )

    assert run.stopping_reason == "FAILURE_REPRODUCED"
    assert run.cycles

    cycle = run.cycles[0]

    # A reproduction record linked to the source finding was persisted.
    assert cycle.reproduction is not None
    assert cycle.reproduction.status.value == "REPRODUCED"
    assert cycle.reproduction.finding_id == (
        execution.finding.finding_id
    )

    row = database.connection.execute(
        """
        SELECT status, finding_id
        FROM reproductions
        WHERE reproduction_id = ?
        """,
        (cycle.reproduction.reproduction_id,),
    ).fetchone()

    assert row is not None
    assert row[0] == "REPRODUCED"
    assert row[1] == execution.finding.finding_id

    # The follow-up experiment is persisted with a workflow marker.
    experiment_row = database.connection.execute(
        """
        SELECT result, context
        FROM experiments
        WHERE experiment_id = ?
        """,
        (cycle.experiment_id,),
    ).fetchone()

    assert experiment_row[0] == "FAIL"
    assert "system_adaptive" in experiment_row[1]

    # The source input was executed once (by the initial assessment) and
    # was not repeated as a follow-up input.
    assert target.inputs.count(SOURCE_INPUT) == 1
    assert len(set(target.inputs)) == len(target.inputs)
