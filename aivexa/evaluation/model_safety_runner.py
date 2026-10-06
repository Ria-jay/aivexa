from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from aivexa.evaluation.adaptive_controller import (
    AdaptiveController,
)
from aivexa.evaluation.finding import (
    Finding,
    FindingBuilder,
)
from aivexa.evaluation.model_safety import (
    ModelSafetySuite,
    SafetyCase,
    SafetyCaseResult,
    SafetyExecution,
    SafetyExecutionStatus,
)
from aivexa.evaluation.models import (
    Evaluation,
    ExperimentResult,
)
from aivexa.evaluation.safety_boundary_engine import (
    SafetyBoundaryEngine,
)
from aivexa.experiments.runner import ExperimentRunner
from aivexa.storage.database import Database


@dataclass(frozen=True)
class ModelSafetyExecution:
    case_result: SafetyCaseResult
    evaluation: Evaluation
    finding: Finding | None
    adaptive_run: Any | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_result": self.case_result.to_dict(),
            "evaluation": {
                "result": self.evaluation.result.value,
                "rationale": self.evaluation.rationale,
                "confidence": self.evaluation.confidence,
                "property_name": self.evaluation.property_name,
                "property_expectation": (
                    self.evaluation.property_expectation
                ),
                "evaluator": self.evaluation.evaluator,
                "evidence": self.evaluation.evidence,
            },
            "finding": (
                {
                    "finding_id": self.finding.finding_id,
                    "assessment_id": self.finding.assessment_id,
                    "experiment_id": self.finding.experiment_id,
                    "title": self.finding.title,
                    "result": self.finding.result.value,
                    "confidence": self.finding.confidence,
                    "affected_properties": (
                        list(self.finding.affected_properties)
                    ),
                    "rationale": self.finding.rationale,
                    "evidence": self.finding.evidence,
                }
                if self.finding is not None
                else None
            ),
            "adaptive_run": (
                self.adaptive_run.to_dict()
                if self.adaptive_run is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ModelSafetyRun:
    target: str
    executions: tuple[ModelSafetyExecution, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "executions": [
                execution.to_dict()
                for execution in self.executions
            ],
            "summary": self.summary,
        }


class ModelSafetyRunner:
    """
    First-class model-safety workflow.

    Flow:

        SafetyCase
            -> Experiment
            -> baseline + variants
            -> SafetyBoundaryEngine
            -> Evaluation
            -> Evidence
            -> Finding
            -> optional adaptive follow-up
            -> persistence

    The runner deliberately does not treat every response change
    as a vulnerability. The SafetyBoundaryEngine determines the
    actual evaluation result.
    """

    def __init__(
        self,
        runner: ExperimentRunner,
        database: Database,
        evaluator: SafetyBoundaryEngine | None = None,
        max_adaptive_cycles: int = 0,
    ):
        if max_adaptive_cycles < 0:
            raise ValueError(
                "max_adaptive_cycles cannot be negative."
            )

        self.runner = runner
        self.database = database
        self.evaluator = (
            evaluator
            if evaluator is not None
            else SafetyBoundaryEngine()
        )
        self.finding_builder = FindingBuilder()
        self.max_adaptive_cycles = max_adaptive_cycles

    def run(
        self,
        cases: tuple[SafetyCase, ...] | None = None,
        generation_options: dict[str, Any] | None = None,
        adaptive: bool = True,
    ) -> ModelSafetyRun:
        selected_cases = (
            cases
            if cases is not None
            else ModelSafetySuite.default_cases()
        )

        executions: list[ModelSafetyExecution] = []

        for case in selected_cases:
            executions.append(
                self._run_case(
                    case=case,
                    generation_options=generation_options,
                    adaptive=adaptive,
                )
            )

        summary = self._summarize(executions)

        return ModelSafetyRun(
            target=self.runner.target.name,
            executions=tuple(executions),
            summary=summary,
        )

    def _run_case(
        self,
        case: SafetyCase,
        generation_options: dict[str, Any] | None,
        adaptive: bool,
    ) -> ModelSafetyExecution:
        experiment_id = (
            f"{case.case_id}-EXPERIMENT-001"
        )

        assessment_id = (
            f"{experiment_id}-ASSESSMENT"
        )

        baseline = self._execute(
            experiment_id=(
                f"{experiment_id}-BASELINE"
            ),
            prompt=case.baseline_prompt,
            generation_options=generation_options,
        )

        variants: list[SafetyExecution] = []

        for index, prompt in enumerate(
            case.variants,
            start=1,
        ):
            variants.append(
                self._execute(
                    experiment_id=(
                        f"{experiment_id}"
                        f"-VARIANT-{index:03d}"
                    ),
                    prompt=prompt,
                    generation_options=generation_options,
                )
            )

        successful_variants = [
            execution.output
            for execution in variants
            if execution.successful
            and execution.output is not None
        ]

        if (
            not baseline.successful
            or not successful_variants
        ):
            evaluation = Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "The safety case could not be evaluated "
                    "because required model executions failed."
                ),
                confidence="Low",
                property_name=case.expected_property,
                property_expectation=(
                    "The defined safety boundary should remain "
                    "preserved under controlled variation."
                ),
                evaluator="aivexa-model-safety-runner",
                evidence={
                    "baseline_successful": (
                        baseline.successful
                    ),
                    "successful_variant_count": (
                        len(successful_variants)
                    ),
                    "variant_count": len(variants),
                    "failed_variant_count": sum(
                        not variant.successful
                        for variant in variants
                    ),
                },
            )
        else:
            assessment = (
                self.evaluator.evaluate_responses(
                    baseline_response=(
                        baseline.output or ""
                    ),
                    variant_responses=successful_variants,
                )
            )

            evaluation = Evaluation(
                result=assessment.result,
                rationale=assessment.rationale,
                confidence=assessment.confidence,
                property_name=case.expected_property,
                property_expectation=(
                    "The defined safety boundary should remain "
                    "preserved under controlled variation."
                ),
                evaluator=(
                    "aivexa-safety-boundary-engine"
                ),
                evidence=dict(
                    assessment.evidence
                ),
            )

        status = self._execution_status(
            baseline=baseline,
            variants=variants,
        )

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        evidence = {
            "safety_case": case.to_dict(),
            "baseline": baseline.to_dict(),
            "variants": [
                variant.to_dict()
                for variant in variants
            ],
            "execution_status": status.value,
            "execution_counts": {
                "baseline": 1,
                "variants": len(variants),
                "successful_variants": (
                    len(successful_variants)
                ),
                "failed_variants": (
                    len(variants)
                    - len(successful_variants)
                ),
            },
        }

        experiment_context = {
            "workflow": "model_safety",
            "target_type": "model",
            "case_id": case.case_id,
            "test_type": case.test_type.value,
            "expected_property": (
                case.expected_property
            ),
            "requires_baseline": case.context.get(
                "requires_baseline",
                False,
            ),
            "requires_conversation_state": (
                case.context.get(
                    "requires_conversation_state",
                    False,
                )
            ),
            "case_context": dict(case.context),
            "safety_execution": evidence,
        }

        observations = list(
            evaluation.evidence.get(
                "observations",
                [],
            )
        )

        observations.append(
            f"Variant executions: {len(variants)}."
        )

        if successful_variants:
            observations.append(
                "At least one variant executed successfully."
            )

        failed_variant_count = sum(
            not variant.successful
            for variant in variants
        )

        if failed_variant_count:
            observations.append(
                "One or more variant executions failed."
            )

        self.database.save_experiment(
            {
                "experiment_id": experiment_id,
                "target": self.runner.target.name,
                "objective": case.objective,
                "hypothesis": case.hypothesis,
                "intervention": case.test_type.value,
                "context": experiment_context,
                "created_at": created_at,
                "input_data": case.baseline_prompt,
                "output_data": (
                    baseline.output or ""
                ),
                "result": evaluation.result.value,
                "rationale": evaluation.rationale,
                "confidence": evaluation.confidence,
                "observations": observations,
                "property_name": (
                    evaluation.property_name
                ),
                "property_expectation": (
                    evaluation.property_expectation
                ),
                "comparison_group": case.case_id,
                "evaluator": evaluation.evaluator,
            }
        )

        self.database.save_evaluation(
            experiment_id=experiment_id,
            assessment_id=assessment_id,
            evaluation=evaluation,
            evidence=evidence,
        )

        finding = (
            self.finding_builder.build_from_evaluation(
                finding_id=(
                    f"{experiment_id}-FINDING"
                ),
                assessment_id=assessment_id,
                experiment_id=experiment_id,
                evaluation=evaluation,
                evidence=evidence,
            )
        )

        if finding is not None:
            self.database.save_finding(finding)

        adaptive_run = None

        if (
            adaptive
            and self.max_adaptive_cycles > 0
            and evaluation.result
            in {
                ExperimentResult.FAIL,
                ExperimentResult.ANOMALY,
            }
            and baseline.successful
            and variants
            and successful_variants
        ):
            adaptive_assessment = (
                self.evaluator.evaluate_responses(
                    baseline_response=(
                        baseline.output or ""
                    ),
                    variant_responses=[
                        successful_variants[0]
                    ],
                )
            )

            adaptive_controller = AdaptiveController(
                runner=self.runner,
                evaluator=self.evaluator,
                database=self.database,
                max_cycles=(
                    self.max_adaptive_cycles
                ),
            )

            adaptive_run = adaptive_controller.run(
                source_experiment_id=experiment_id,
                target=self.runner.target.name,
                baseline_prompt=(
                    case.baseline_prompt
                ),
                initial_assessment=(
                    adaptive_assessment
                ),
                adaptive_prompt=case.variants[0],
                generation_options=(
                    generation_options
                ),
            )

        case_result = SafetyCaseResult(
            case=case,
            baseline=baseline,
            variants=tuple(variants),
            result=evaluation.result.value,
            confidence=evaluation.confidence,
            rationale=evaluation.rationale,
            evidence=evidence,
            status=status,
        )

        return ModelSafetyExecution(
            case_result=case_result,
            evaluation=evaluation,
            finding=finding,
            adaptive_run=adaptive_run,
        )

    def _execute(
        self,
        experiment_id: str,
        prompt: str,
        generation_options: dict[str, Any] | None,
    ) -> SafetyExecution:
        try:
            evidence = self.runner.run(
                experiment_id=experiment_id,
                prompt=prompt,
                generation_options=(
                    generation_options
                ),
            )

            return SafetyExecution(
                prompt=prompt,
                output=evidence.output_data,
                error=None,
                status=(
                    SafetyExecutionStatus.COMPLETED.value
                ),
            )

        except Exception as exc:
            return SafetyExecution(
                prompt=prompt,
                output=None,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                status=(
                    SafetyExecutionStatus.EXECUTION_ERROR.value
                ),
            )

    @staticmethod
    def _execution_status(
        baseline: SafetyExecution,
        variants: list[SafetyExecution],
    ) -> SafetyExecutionStatus:
        if not baseline.successful:
            return SafetyExecutionStatus.EXECUTION_ERROR

        if all(
            variant.successful
            for variant in variants
        ):
            return SafetyExecutionStatus.COMPLETED

        return SafetyExecutionStatus.PARTIAL_EXECUTION

    @staticmethod
    def _summarize(
        executions: list[ModelSafetyExecution],
    ) -> dict[str, Any]:
        counts: dict[str, int] = {}
        statuses: dict[str, int] = {}
        findings = 0

        for execution in executions:
            result = (
                execution.evaluation.result.value
            )

            counts[result] = (
                counts.get(result, 0) + 1
            )

            status = (
                execution.case_result.status.value
            )

            statuses[status] = (
                statuses.get(status, 0) + 1
            )

            if execution.finding is not None:
                findings += 1

        return {
            "total_cases": len(executions),
            "result_counts": counts,
            "execution_statuses": statuses,
            "findings": findings,
            "evidence_scope": (
                "Controlled model-safety cases executed "
                "through the AIVEXA experiment workflow."
            ),
            "warning": (
                "Results describe tested properties only and "
                "do not establish universal model safety."
            ),
        }
