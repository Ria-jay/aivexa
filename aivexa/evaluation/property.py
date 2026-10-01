from aivexa.evaluation.content import ContentEvaluation
from aivexa.evaluation.models import Evaluation, ExperimentResult


def evaluate_content_property(
    content: ContentEvaluation,
    property_name: str,
    property_expectation: str,
) -> Evaluation:
    if content.preserved:
        return Evaluation(
            result=ExperimentResult.PASS,
            rationale=content.rationale,
            confidence="Medium",
            property_name=property_name,
            property_expectation=property_expectation,
        )

    return Evaluation(
        result=ExperimentResult.FAIL,
        rationale=content.rationale,
        confidence="Medium",
        property_name=property_name,
        property_expectation=property_expectation,
    )
