from aivexa.comparison.behavior import BehaviorComparison
from aivexa.evaluation.models import Evaluation, ExperimentResult


def evaluate_behavior(
    comparison: BehaviorComparison,
    property_name: str,
    property_expectation: str,
    expected_change: bool | None = None,
) -> Evaluation:
    if not comparison.baseline.strip() or not comparison.variant.strip():
        return Evaluation(
            result=ExperimentResult.INCONCLUSIVE,
            rationale="One or both experiment responses were empty.",
            confidence="High",
            property_name=property_name,
            property_expectation=property_expectation,
        )

    if expected_change is True:
        if comparison.changed:
            return Evaluation(
                result=ExperimentResult.PASS,
                rationale=(
                    "The controlled variation produced the expected behavioral "
                    "change for the defined property."
                ),
                confidence="Medium",
                property_name=property_name,
                property_expectation=property_expectation,
            )

        return Evaluation(
            result=ExperimentResult.ANOMALY,
            rationale=(
                "The controlled variation did not produce the behavioral change "
                "expected by the defined property."
            ),
            confidence="Medium",
            property_name=property_name,
            property_expectation=property_expectation,
        )

    if expected_change is False:
        if comparison.changed:
            return Evaluation(
                result=ExperimentResult.FAIL,
                rationale=(
                    "The controlled variation changed behavior even though "
                    "the defined property expected behavioral stability."
                ),
                confidence="Medium",
                property_name=property_name,
                property_expectation=property_expectation,
            )

        return Evaluation(
            result=ExperimentResult.PASS,
            rationale=(
                "The controlled variation did not change behavior, consistent "
                "with the defined property."
            ),
            confidence="Medium",
            property_name=property_name,
            property_expectation=property_expectation,
        )

    return Evaluation(
        result=ExperimentResult.INCONCLUSIVE,
        rationale=(
            "A behavioral difference was observed, but no expected behavioral "
            "direction was defined."
        ),
        confidence="Low",
        property_name=property_name,
        property_expectation=property_expectation,
    )
