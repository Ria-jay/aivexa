from __future__ import annotations

from aivexa.research.accuracy import (
    EvaluationAccuracyCalculator,
    LabeledEvaluation,
)


def test_accuracy_distinguishes_fp_and_fn():
    calculator = EvaluationAccuracyCalculator()

    result = calculator.calculate(
        [
            LabeledEvaluation(
                case_id="tp",
                expected_result="FAIL",
                actual_result="FAIL",
            ),
            LabeledEvaluation(
                case_id="tn",
                expected_result="PASS",
                actual_result="PASS",
            ),
            LabeledEvaluation(
                case_id="fp",
                expected_result="PASS",
                actual_result="FAIL",
            ),
            LabeledEvaluation(
                case_id="fn",
                expected_result="FAIL",
                actual_result="PASS",
            ),
        ]
    )

    assert result.true_positive == 1
    assert result.true_negative == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.accuracy == 0.5


def test_accuracy_surfaces_non_definitive_outcomes():
    calculator = EvaluationAccuracyCalculator()

    result = calculator.calculate(
        [
            LabeledEvaluation(
                case_id="a",
                expected_result="FAIL",
                actual_result="INCONCLUSIVE",
            ),
            LabeledEvaluation(
                case_id="b",
                expected_result="PASS",
                actual_result="ANOMALY",
            ),
            LabeledEvaluation(
                case_id="c",
                expected_result="PASS",
                actual_result="NOT_APPLICABLE",
            ),
            LabeledEvaluation(
                case_id="d",
                expected_result="FAIL",
                actual_result="INCONCLUSIVE",
                execution_status="EXECUTION_ERROR",
            ),
        ]
    )

    assert result.inconclusive == 1
    assert result.anomaly == 1
    assert result.not_applicable == 1
    assert result.execution_failure == 1
    assert result.precision is None
    assert result.recall is None


def test_accuracy_no_definitive_labels_has_no_rates():
    calculator = EvaluationAccuracyCalculator()

    result = calculator.calculate([])

    assert result.total == 0
    assert result.accuracy is None
