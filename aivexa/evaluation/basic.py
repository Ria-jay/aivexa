from aivexa.comparison.behavior import BehaviorComparison
from aivexa.evaluation.models import Evaluation, ExperimentResult


def evaluate_behavior(comparison: BehaviorComparison) -> Evaluation:
    if not comparison.baseline.strip() or not comparison.variant.strip():
        return Evaluation(
            result=ExperimentResult.INCONCLUSIVE,
            rationale="One or both experiment responses were empty.",
            confidence="High",
        )

    if comparison.changed:
        return Evaluation(
            result=ExperimentResult.ANOMALY,
            rationale=(
                "The controlled variation produced a measurable behavioral "
                "difference from the baseline."
            ),
            confidence="Medium",
        )

    return Evaluation(
        result=ExperimentResult.PASS,
        rationale=(
            "The controlled variation did not produce a measurable "
            "difference from the baseline."
        ),
        confidence="Medium",
    )
