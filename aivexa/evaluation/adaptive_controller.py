from dataclasses import dataclass
from typing import Any

from aivexa.evaluation.adaptive import AdaptiveProbePlanner, AdaptiveProbe
from aivexa.evaluation.adaptive_execution import (
    AdaptiveExperimentExecutor,
    AdaptiveExecution,
)
from aivexa.evaluation.adaptive_persistence import (
    AdaptiveExperimentPersistence,
)
from aivexa.evaluation.adaptive_runner import (
    AdaptiveExperimentBuilder,
)
from aivexa.evaluation.safety_boundary_engine import (
    SafetyBoundaryEngine,
)
from aivexa.experiments.runner import ExperimentRunner


@dataclass(frozen=True)
class AdaptiveCycle:
    cycle_number: int
    source_experiment_id: str
    probe: AdaptiveProbe
    execution: AdaptiveExecution

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_number": self.cycle_number,
            "source_experiment_id": self.source_experiment_id,
            "probe": self.probe.to_dict(),
            "execution": self.execution.to_dict(),
        }


@dataclass(frozen=True)
class AdaptiveRun:
    source_experiment_id: str
    cycles: tuple[AdaptiveCycle, ...]
    stopping_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_experiment_id": self.source_experiment_id,
            "cycles": [
                cycle.to_dict()
                for cycle in self.cycles
            ],
            "stopping_reason": self.stopping_reason,
        }


class AdaptiveController:
    """
    Runs a bounded adaptive evaluation loop.

    Each cycle produces a unique probe and experiment.
    Each new cycle is derived from the immediately preceding
    experiment result.
    """

    def __init__(
        self,
        runner: ExperimentRunner,
        evaluator: SafetyBoundaryEngine,
        database,
        max_cycles: int = 2,
    ):
        if max_cycles < 1:
            raise ValueError("max_cycles must be at least 1.")

        self.runner = runner
        self.evaluator = evaluator
        self.database = database
        self.max_cycles = max_cycles

        self.planner = AdaptiveProbePlanner()
        self.builder = AdaptiveExperimentBuilder()
        self.executor = AdaptiveExperimentExecutor(
            runner=runner,
            evaluator=evaluator,
        )
        self.persistence = AdaptiveExperimentPersistence(
            database
        )

    def run(
        self,
        source_experiment_id: str,
        target: str,
        baseline_prompt: str,
        initial_assessment,
        adaptive_prompt: str,
        generation_options: dict | None = None,
    ) -> AdaptiveRun:
        cycles: list[AdaptiveCycle] = []
        assessment = initial_assessment

        current_source_id = source_experiment_id

        for cycle_number in range(1, self.max_cycles + 1):
            planned_probe = self.planner.plan(
                assessment,
                current_source_id,
            )

            probe = AdaptiveProbe(
                probe_id=(
                    f"{source_experiment_id}"
                    f"-ADAPTIVE-{cycle_number:03d}"
                ),
                objective=planned_probe.objective,
                hypothesis=planned_probe.hypothesis,
                intervention_type=planned_probe.intervention_type,
                reason=planned_probe.reason,
                priority=planned_probe.priority,
                expected_property=planned_probe.expected_property,
                context={
                    **planned_probe.context,
                    "cycle_number": cycle_number,
                    "source_experiment_id": current_source_id,
                },
            )

            adaptive_experiment = self.builder.build(
                probe=probe,
                target=target,
                prompt=adaptive_prompt,
                source_experiment_id=current_source_id,
            )

            execution = self.executor.execute(
                experiment=adaptive_experiment.experiment,
                probe=probe,
                baseline_prompt=baseline_prompt,
                adaptive_prompt=adaptive_prompt,
                generation_options=generation_options,
            )

            self.persistence.save(execution)

            cycle = AdaptiveCycle(
                cycle_number=cycle_number,
                source_experiment_id=current_source_id,
                probe=probe,
                execution=execution,
            )

            cycles.append(cycle)

            if execution.evaluation.result.value == "FAIL":
                return AdaptiveRun(
                    source_experiment_id=source_experiment_id,
                    cycles=tuple(cycles),
                    stopping_reason="FAILURE_REPRODUCED",
                )

            assessment = self.evaluator.evaluate_responses(
                baseline_response=execution.baseline_output,
                variant_responses=[
                    execution.adaptive_output,
                ],
            )

            current_source_id = execution.experiment.experiment_id

        return AdaptiveRun(
            source_experiment_id=source_experiment_id,
            cycles=tuple(cycles),
            stopping_reason="MAX_CYCLES_REACHED",
        )
