from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LabeledEvaluation:
    """
    One evaluator outcome with a ground-truth label.

    ``expected_result`` is the ground truth for the tested property.
    ``actual_result`` is what the evaluator produced. Either may be
    FAIL, PASS, ANOMALY, INCONCLUSIVE, or NOT_APPLICABLE.
    """

    case_id: str
    expected_result: str
    actual_result: str
    execution_status: str = "COMPLETED"
    property_name: str | None = None
    evaluator: str | None = None


@dataclass(frozen=True)
class EvaluationAccuracy:
    """
    Confusion-matrix style accuracy metrics over labelled evaluations.

    Only definitive labels (FAIL as "positive", PASS as "negative") are
    used for precision/recall. ANOMALY, INCONCLUSIVE, NOT_APPLICABLE,
    and execution failures are surfaced explicitly rather than being
    silently folded into accuracy.
    """

    total: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    anomaly: int
    inconclusive: int
    not_applicable: int
    execution_failure: int
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rate(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


class EvaluationAccuracyCalculator:
    """
    Measures evaluator false-positive and false-negative behaviour
    against ground truth where it exists.

    A heuristic evaluator score is never treated as ground truth; this
    calculator requires explicit labels supplied by the caller.
    """

    def calculate(
        self,
        labelled: list[LabeledEvaluation],
    ) -> EvaluationAccuracy:
        tp = fp = fn = tn = 0
        anomaly = inconclusive = not_applicable = 0
        execution_failure = 0

        details: list[dict[str, Any]] = []

        for item in labelled:
            expected = item.expected_result
            actual = item.actual_result

            if item.execution_status not in {
                "COMPLETED",
                "PARTIAL_EXECUTION",
            }:
                execution_failure += 1
                details.append(
                    {
                        **asdict(item),
                        "category": "execution_failure",
                    }
                )
                continue

            if actual == "ANOMALY":
                anomaly += 1
                details.append(
                    {**asdict(item), "category": "anomaly"}
                )
                continue

            if actual == "NOT_APPLICABLE":
                not_applicable += 1
                details.append(
                    {
                        **asdict(item),
                        "category": "not_applicable",
                    }
                )
                continue

            if actual == "INCONCLUSIVE":
                inconclusive += 1
                details.append(
                    {
                        **asdict(item),
                        "category": "inconclusive",
                    }
                )
                continue

            if expected == "FAIL":
                if actual == "FAIL":
                    tp += 1
                    category = "true_positive"
                else:
                    fn += 1
                    category = "false_negative"
            elif expected == "PASS":
                if actual == "FAIL":
                    fp += 1
                    category = "false_positive"
                else:
                    tn += 1
                    category = "true_negative"
            else:
                category = "unlabelled"

            details.append(
                {**asdict(item), "category": category}
            )

        precision = _rate(tp, tp + fp)
        recall = _rate(tp, tp + fn)
        accuracy = _rate(tp + tn, tp + tn + fp + fn)

        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = round(
                2 * precision * recall
                / (precision + recall),
                4,
            )

        return EvaluationAccuracy(
            total=len(labelled),
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            true_negative=tn,
            anomaly=anomaly,
            inconclusive=inconclusive,
            not_applicable=not_applicable,
            execution_failure=execution_failure,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            false_positive_rate=_rate(fp, fp + tn),
            false_negative_rate=_rate(fn, fn + tp),
            details=details,
        )
