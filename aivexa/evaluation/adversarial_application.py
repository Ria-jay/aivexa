from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from aivexa.evaluation.adversarial import (
    AdversarialCase,
    AdversarialCaseLibrary,
)
from aivexa.evaluation.adversarial_profile import (
    AdversarialCaseSelector,
    AdversarialTargetProfile,
)
from aivexa.evaluation.adversarial_properties import (
    AdversarialPropertyRegistry,
)
from aivexa.evaluation.finding import Finding
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evidence.system import SystemInteraction
from aivexa.storage.database import Database
from aivexa.targets.application import AIApplicationTarget


@dataclass(frozen=True)
class ApplicationAdversarialCaseResult:
    case: AdversarialCase
    experiment_id: str
    baseline_interaction: SystemInteraction | None
    variant_interactions: tuple[SystemInteraction, ...]
    evaluation: Evaluation
    finding: Finding | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "experiment_id": self.experiment_id,
            "baseline_interaction": (
                None
                if self.baseline_interaction is None
                else self._interaction_dict(
                    self.baseline_interaction
                )
            ),
            "variant_interactions": [
                self._interaction_dict(interaction)
                for interaction in self.variant_interactions
            ],
            "evaluation": {
                "result": self.evaluation.result.value,
                "confidence": self.evaluation.confidence,
                "rationale": self.evaluation.rationale,
                "property_name": self.evaluation.property_name,
                "property_expectation": (
                    self.evaluation.property_expectation
                ),
                "evaluator": self.evaluation.evaluator,
                "evidence": self.evaluation.evidence,
            },
            "finding": (
                None
                if self.finding is None
                else {
                    "finding_id": self.finding.finding_id,
                    "title": self.finding.title,
                    "result": self.finding.result.value,
                    "confidence": self.finding.confidence,
                }
            ),
        }

    @staticmethod
    def _interaction_dict(
        interaction: SystemInteraction,
    ) -> dict[str, Any]:
        return {
            "experiment_id": interaction.experiment_id,
            "user_input": interaction.user_input,
            "system_output": interaction.system_output,
            "context": dict(interaction.context),
            "retrieved_context": list(
                interaction.retrieved_context
            ),
            "tool_calls": list(interaction.tool_calls),
            "authorization_context": dict(
                interaction.authorization_context
            ),
            "observations": list(interaction.observations),
            "metadata": dict(interaction.metadata),
            "observed_at": interaction.observed_at,
        }


class ApplicationAdversarialWorkflow:
    """
    Runs adversarial cases against an AI application target.

    The workflow delegates security-property judgments to the
    existing AIVEXA evaluator architecture.
    """

    def __init__(
        self,
        target: AIApplicationTarget,
        database: Database,
    ):
        self.target = target
        self.database = database
        self.selector = AdversarialCaseSelector()
        self.properties = AdversarialPropertyRegistry()

    def run(
        self,
        profile: AdversarialTargetProfile,
        cases: tuple[AdversarialCase, ...] | None = None,
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected_cases = (
            cases
            if cases is not None
            else tuple(AdversarialCaseLibrary.default_cases())
        )

        applicable = self.selector.select(
            selected_cases,
            profile,
        )

        applicable_ids = {
            case.case_id
            for case in applicable
        }

        skipped = [
            case.case_id
            for case in selected_cases
            if case.case_id not in applicable_ids
        ]

        results = [
            self._run_case(
                case=case,
                profile=profile,
                context=context,
                authorization_context=authorization_context,
            )
            for case in applicable
        ]

        return {
            "target_profile": profile.to_dict(),
            "results": [
                result.to_dict()
                for result in results
            ],
            "skipped_cases": skipped,
            "summary": self._summary(
                results,
                skipped,
            ),
        }

    def _run_case(
        self,
        case: AdversarialCase,
        profile: AdversarialTargetProfile,
        context: dict[str, Any] | None,
        authorization_context: dict[str, Any] | None,
    ) -> ApplicationAdversarialCaseResult:
        root_id = (
            f"{case.case_id}-{uuid4().hex[:10]}"
        )

        baseline = self._execute(
            experiment_id=f"{root_id}-BASELINE",
            user_input=case.baseline_intervention,
            context=context,
            authorization_context=authorization_context,
        )

        variants: list[SystemInteraction] = []

        for variant in case.variants:
            interaction = self._execute(
                experiment_id=(
                    f"{root_id}-{variant.variant_id}"
                ),
                user_input=variant.intervention,
                context={
                    **(context or {}),
                    **variant.context,
                },
                authorization_context=authorization_context,
            )

            if interaction is not None:
                variants.append(interaction)

        if baseline is None:
            evaluation = self._inconclusive(
                case,
                "Baseline interaction could not be executed.",
            )
        elif len(variants) != len(case.variants):
            evaluation = self._inconclusive(
                case,
                "One or more adversarial variants could not be executed.",
            )
        else:
            evaluations = [
                self.properties.evaluate_interaction(
                    case=case,
                    interaction=interaction,
                    baseline_output=baseline.system_output,
                )
                for interaction in variants
            ]

            evaluation = self._aggregate(
                case=case,
                evaluations=evaluations,
                baseline=baseline,
                variants=variants,
            )

        self._persist(
            root_id=root_id,
            case=case,
            profile=profile,
            baseline=baseline,
            variants=variants,
            evaluation=evaluation,
        )

        finding = self._build_finding(
            root_id=root_id,
            case=case,
            evaluation=evaluation,
        )

        if finding is not None:
            self.database.save_finding(finding)

        return ApplicationAdversarialCaseResult(
            case=case,
            experiment_id=root_id,
            baseline_interaction=baseline,
            variant_interactions=tuple(variants),
            evaluation=evaluation,
            finding=finding,
        )

    def _execute(
        self,
        experiment_id: str,
        user_input: str,
        context: dict[str, Any] | None,
        authorization_context: dict[str, Any] | None,
    ) -> SystemInteraction | None:
        try:
            return self.target.interact(
                experiment_id=experiment_id,
                user_input=user_input,
                context=context,
                authorization_context=authorization_context,
            )
        except Exception:
            return None

    @staticmethod
    def _aggregate(
        case: AdversarialCase,
        evaluations: list[Evaluation],
        baseline: SystemInteraction,
        variants: list[SystemInteraction],
    ) -> Evaluation:
        if not evaluations:
            return ApplicationAdversarialWorkflow._inconclusive(
                case,
                "No property evaluation was produced.",
            )

        results = [
            evaluation.result
            for evaluation in evaluations
        ]

        if ExperimentResult.FAIL in results:
            result = ExperimentResult.FAIL
        elif ExperimentResult.ANOMALY in results:
            result = ExperimentResult.ANOMALY
        elif all(
            result == ExperimentResult.PASS
            for result in results
        ):
            result = ExperimentResult.PASS
        else:
            result = ExperimentResult.INCONCLUSIVE

        confidence_rank = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
        }

        confidence = min(
            (
                evaluation.confidence
                for evaluation in evaluations
            ),
            key=lambda value: confidence_rank.get(
                value,
                1,
            ),
        )

        return Evaluation(
            result=result,
            rationale=(
                "Adversarial variants were evaluated using "
                "the registered AIVEXA security property "
                "evaluator."
            ),
            confidence=confidence,
            property_name=case.property_name,
            property_expectation=case.property_expectation,
            evaluator="aivexa-adversarial-application",
            evidence={
                "baseline": {
                    "experiment_id": baseline.experiment_id,
                    "output": baseline.system_output,
                    "retrieved_context": list(
                        baseline.retrieved_context
                    ),
                    "tool_calls": list(
                        baseline.tool_calls
                    ),
                    "authorization_context": dict(
                        baseline.authorization_context
                    ),
                    "observations": list(
                        baseline.observations
                    ),
                },
                "variants": [
                    {
                        "experiment_id": interaction.experiment_id,
                        "output": interaction.system_output,
                        "retrieved_context": list(
                            interaction.retrieved_context
                        ),
                        "tool_calls": list(
                            interaction.tool_calls
                        ),
                        "authorization_context": dict(
                            interaction.authorization_context
                        ),
                        "observations": list(
                            interaction.observations
                        ),
                    }
                    for interaction in variants
                ],
                "individual_evaluations": [
                    {
                        "result": evaluation.result.value,
                        "confidence": evaluation.confidence,
                        "rationale": evaluation.rationale,
                        "property_name": evaluation.property_name,
                        "evaluator": evaluation.evaluator,
                        "evidence": evaluation.evidence,
                    }
                    for evaluation in evaluations
                ],
            },
        )

    def _persist(
        self,
        root_id: str,
        case: AdversarialCase,
        profile: AdversarialTargetProfile,
        baseline: SystemInteraction | None,
        variants: list[SystemInteraction],
        evaluation: Evaluation,
    ) -> None:
        self.database.save_experiment(
            {
                "experiment_id": root_id,
                "target": profile.target_name,
                "objective": case.objective,
                "hypothesis": case.hypothesis,
                "intervention": "ADVERSARIAL_VARIATION",
                "context": {
                    "adversarial": True,
                    "case": case.to_dict(),
                    "target_profile": profile.to_dict(),
                },
                "created_at": (
                    baseline.observed_at
                    if baseline is not None
                    else ""
                ),
                "input_data": case.baseline_intervention,
                "output_data": (
                    baseline.system_output
                    if baseline is not None
                    else ""
                ),
                "result": evaluation.result.value,
                "rationale": evaluation.rationale,
                "confidence": evaluation.confidence,
                "observations": (
                    list(baseline.observations)
                    if baseline is not None
                    else []
                ),
                "property_name": evaluation.property_name,
                "property_expectation": (
                    evaluation.property_expectation
                ),
                "evaluator": evaluation.evaluator,
                "reproducibility": {
                    "variant_count": len(variants),
                },
            }
        )

    @staticmethod
    def _build_finding(
        root_id: str,
        case: AdversarialCase,
        evaluation: Evaluation,
    ) -> Finding | None:
        if evaluation.result not in {
            ExperimentResult.FAIL,
            ExperimentResult.ANOMALY,
        }:
            return None

        return Finding(
            finding_id=f"FINDING-{uuid4().hex[:12]}",
            assessment_id=f"ADV-ASSESS-{root_id}",
            experiment_id=root_id,
            title=(
                f"Adversarial evaluation: "
                f"{case.property_name}"
            ),
            result=evaluation.result,
            confidence=evaluation.confidence,
            affected_properties=[case.property_name],
            rationale=evaluation.rationale,
            evidence=evaluation.evidence,
        )

    @staticmethod
    def _inconclusive(
        case: AdversarialCase,
        rationale: str,
    ) -> Evaluation:
        return Evaluation(
            result=ExperimentResult.INCONCLUSIVE,
            rationale=rationale,
            confidence="Low",
            property_name=case.property_name,
            property_expectation=case.property_expectation,
            evaluator="aivexa-adversarial-application",
            evidence={
                "execution_incomplete": True,
            },
        )

    @staticmethod
    def _summary(
        results: list[ApplicationAdversarialCaseResult],
        skipped: list[str],
    ) -> dict[str, Any]:
        counts = {
            result.value: 0
            for result in ExperimentResult
        }

        findings = 0

        for case_result in results:
            counts[
                case_result.evaluation.result.value
            ] += 1

            if case_result.finding is not None:
                findings += 1

        return {
            **counts,
            "executed": len(results),
            "skipped": len(skipped),
            "findings": findings,
        }
