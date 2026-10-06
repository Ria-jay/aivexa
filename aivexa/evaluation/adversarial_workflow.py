from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from aivexa.evaluation.adversarial import (
    AdversarialCase,
    AdversarialCaseLibrary,
    AdversarialVariant,
)
from aivexa.evaluation.adversarial_engine import (
    AdversarialEvaluationEngine,
    AdversarialObservation,
)
from aivexa.evaluation.adversarial_profile import (
    AdversarialCaseSelector,
    AdversarialTargetProfile,
)
from aivexa.evaluation.adversarial_properties import (
    build_default_property_registry,
)
from aivexa.evaluation.finding import Finding
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evidence.models import Evidence
from aivexa.experiments.runner import ExperimentRunner
from aivexa.storage.database import Database


@dataclass(frozen=True)
class AdversarialCaseResult:
    case: AdversarialCase
    experiment_id: str
    evaluation: Evaluation
    baseline: Evidence | None
    variants: tuple[Evidence, ...]
    observations: tuple[AdversarialObservation, ...]
    finding: Finding | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "experiment_id": self.experiment_id,
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
            "baseline": (
                None
                if self.baseline is None
                else self.baseline.output_data
            ),
            "variants": [
                variant.output_data
                for variant in self.variants
            ],
            "observations": [
                observation.to_dict()
                for observation in self.observations
            ],
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


@dataclass(frozen=True)
class AdversarialRun:
    target_profile: AdversarialTargetProfile
    results: tuple[AdversarialCaseResult, ...]
    skipped_cases: tuple[str, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_profile": self.target_profile.to_dict(),
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "skipped_cases": list(self.skipped_cases),
            "summary": self.summary,
        }


class AdversarialWorkflow:
    """
    Enterprise-oriented adversarial evaluation workflow.

    The workflow:
        profile
        -> select applicable tests
        -> baseline
        -> controlled adversarial variants
        -> property evaluation
        -> evidence
        -> finding when justified

    It does not assume that every adversarial behavior is a
    vulnerability.
    """

    def __init__(
        self,
        runner: ExperimentRunner,
        database: Database,
    ):
        self.runner = runner
        self.database = database
        self.selector = AdversarialCaseSelector()
        self.engine = AdversarialEvaluationEngine()
        self.properties = build_default_property_registry()

    def run(
        self,
        profile: AdversarialTargetProfile,
        cases: tuple[AdversarialCase, ...] | None = None,
        generation_options: dict | None = None,
    ) -> AdversarialRun:
        selected_cases = (
            cases
            if cases is not None
            else AdversarialCaseLibrary.default_cases()
        )

        applicable = self.selector.select(
            selected_cases,
            profile,
        )

        applicable_ids = {
            case.case_id
            for case in applicable
        }

        skipped = tuple(
            case.case_id
            for case in selected_cases
            if case.case_id not in applicable_ids
        )

        results = [
            self._run_case(
                case=case,
                profile=profile,
                generation_options=generation_options,
            )
            for case in applicable
        ]

        return AdversarialRun(
            target_profile=profile,
            results=tuple(results),
            skipped_cases=skipped,
            summary=self._summary(
                results=results,
                skipped=skipped,
            ),
        )

    def _run_case(
        self,
        case: AdversarialCase,
        profile: AdversarialTargetProfile,
        generation_options: dict | None,
    ) -> AdversarialCaseResult:
        experiment_id = (
            f"{case.case_id}-{uuid4().hex[:10]}"
        )

        baseline = self._execute(
            experiment_id=(
                f"{experiment_id}-BASELINE"
            ),
            intervention=case.baseline_intervention,
            generation_options=generation_options,
        )

        variants: list[Evidence] = []
        observations: list[AdversarialObservation] = []

        for variant in case.variants:
            evidence = self._execute(
                experiment_id=(
                    f"{experiment_id}-{variant.variant_id}"
                ),
                intervention=variant.intervention,
                generation_options=generation_options,
            )

            if evidence is None:
                continue

            variants.append(evidence)

            observations.append(
                AdversarialObservation(
                    variant_id=variant.variant_id,
                    baseline_output=(
                        baseline.output_data
                        if baseline is not None
                        else ""
                    ),
                    variant_output=evidence.output_data,
                    context=variant.context,
                    observations=[
                        (
                            "Baseline and adversarial outputs "
                            "were captured."
                        )
                    ],
                )
            )

        evaluation = self.engine.evaluate(
            case=case,
            observations=observations,
            property_evaluator=(
                self.properties.get(
                    case.property_name
                )
            ),
        )

        if (
            baseline is None
            or len(variants) < len(case.variants)
        ):
            evaluation = Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    f"{evaluation.rationale} "
                    "One or more required target interactions "
                    "could not be executed."
                ),
                confidence="Low",
                property_name=evaluation.property_name,
                property_expectation=(
                    evaluation.property_expectation
                ),
                evaluator=evaluation.evaluator,
                evidence={
                    **evaluation.evidence,
                    "execution_incomplete": True,
                    "expected_variants": len(
                        case.variants
                    ),
                    "completed_variants": len(
                        variants
                    ),
                },
            )

        self._persist(
            experiment_id=experiment_id,
            case=case,
            profile=profile,
            baseline=baseline,
            variants=variants,
            evaluation=evaluation,
        )

        finding = self._finding(
            experiment_id=experiment_id,
            case=case,
            profile=profile,
            baseline=baseline,
            variants=variants,
            evaluation=evaluation,
        )

        if finding is not None:
            self.database.save_finding(finding)

        return AdversarialCaseResult(
            case=case,
            experiment_id=experiment_id,
            evaluation=evaluation,
            baseline=baseline,
            variants=tuple(variants),
            observations=tuple(observations),
            finding=finding,
        )

    def _execute(
        self,
        experiment_id: str,
        intervention: str,
        generation_options: dict | None,
    ) -> Evidence | None:
        try:
            return self.runner.run(
                experiment_id=experiment_id,
                prompt=intervention,
                generation_options=generation_options,
            )
        except Exception:
            return None

    def _persist(
        self,
        experiment_id: str,
        case: AdversarialCase,
        profile: AdversarialTargetProfile,
        baseline: Evidence | None,
        variants: list[Evidence],
        evaluation: Evaluation,
    ) -> None:
        context = {
            "adversarial": True,
            "case": case.to_dict(),
            "target_profile": profile.to_dict(),
            "baseline": (
                None
                if baseline is None
                else {
                    "input": baseline.input_data,
                    "output": baseline.output_data,
                }
            ),
            "variants": [
                {
                    "input": evidence.input_data,
                    "output": evidence.output_data,
                }
                for evidence in variants
            ],
        }

        self.database.save_experiment(
            {
                "experiment_id": experiment_id,
                "target": profile.target_name,
                "objective": case.objective,
                "hypothesis": case.hypothesis,
                "intervention": "ADVERSARIAL_VARIATION",
                "context": context,
                "created_at": self._timestamp(),
                "input_data": json.dumps(
                    {
                        "baseline": (
                            case.baseline_intervention
                        ),
                        "variants": [
                            variant.intervention
                            for variant in case.variants
                        ],
                    }
                ),
                "output_data": (
                    baseline.output_data
                    if baseline is not None
                    else ""
                ),
                "result": evaluation.result.value,
                "rationale": evaluation.rationale,
                "confidence": evaluation.confidence,
                "observations": [
                    (
                        f"Completed {len(variants)} of "
                        f"{len(case.variants)} adversarial variants."
                    )
                ],
                "property_name": evaluation.property_name,
                "property_expectation": (
                    evaluation.property_expectation
                ),
                "comparison_group": case.case_id,
                "evaluator": evaluation.evaluator,
                "reproducibility": {
                    "case_id": case.case_id,
                    "target": profile.target_name,
                },
            }
        )

        class AssessmentRecord:
            pass

        assessment = AssessmentRecord()
        assessment.assessment_id = (
            f"{experiment_id}-ASSESSMENT"
        )
        assessment.experiment_id = experiment_id
        assessment.result = evaluation.result
        assessment.confidence = evaluation.confidence
        assessment.rationale = evaluation.rationale
        assessment.evidence = evaluation.evidence
        assessment.evaluations = (evaluation,)

        self.database.save_assessment(assessment)

    def _finding(
        self,
        experiment_id: str,
        case: AdversarialCase,
        profile: AdversarialTargetProfile,
        baseline: Evidence | None,
        variants: list[Evidence],
        evaluation: Evaluation,
    ) -> Finding | None:
        if evaluation.result not in {
            ExperimentResult.FAIL,
            ExperimentResult.ANOMALY,
        }:
            return None

        title = (
            "Adversarial security property failure"
            if evaluation.result == ExperimentResult.FAIL
            else "Adversarial security property anomaly"
        )

        return Finding(
            finding_id=(
                f"ADV-{uuid4().hex[:12]}"
            ),
            assessment_id=(
                f"{experiment_id}-ASSESSMENT"
            ),
            experiment_id=experiment_id,
            title=title,
            result=evaluation.result,
            confidence=evaluation.confidence,
            affected_properties=[
                evaluation.property_name
            ],
            rationale=evaluation.rationale,
            evidence={
                "target_profile": profile.to_dict(),
                "case": case.to_dict(),
                "evaluation": evaluation.evidence,
                "baseline": (
                    None
                    if baseline is None
                    else baseline.output_data
                ),
                "variants": [
                    evidence.output_data
                    for evidence in variants
                ],
            },
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _summary(
        results: list[AdversarialCaseResult],
        skipped: tuple[str, ...],
    ) -> dict[str, Any]:
        counts: dict[str, int] = {}

        for result in results:
            name = result.evaluation.result.value
            counts[name] = counts.get(name, 0) + 1

        return {
            "executed": len(results),
            "skipped": len(skipped),
            "result_counts": counts,
            "finding_count": sum(
                result.finding is not None
                for result in results
            ),
        }
