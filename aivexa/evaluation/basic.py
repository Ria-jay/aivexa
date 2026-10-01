from aivexa.evaluation.models import Evaluation, ExperimentResult


def evaluate_response(response: str) -> Evaluation:
    if not response.strip():
        return Evaluation(
            result=ExperimentResult.INCONCLUSIVE,
            rationale="The target returned an empty response.",
            confidence="High",
        )

    return Evaluation(
        result=ExperimentResult.PASS,
        rationale="The target produced a non-empty response.",
        confidence="Low",
    )
