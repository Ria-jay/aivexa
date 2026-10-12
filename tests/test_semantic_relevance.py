from __future__ import annotations

from aivexa.comparison.behavior import compare_responses
from aivexa.evaluation.models import ExperimentResult
from aivexa.evaluation.semantic_relevance import (
    SemanticEvaluatorUnavailable,
    SemanticRelevanceEvaluator,
    combine_with_deterministic,
)


class FakeClient:
    """Deterministic embedding stand-in for tests."""

    def __init__(self, relevant: bool):
        self.relevant = relevant

    def embed(self, texts):
        vectors = []

        for text in texts:
            if "hazard" in text.lower() or (
                self.relevant and text == texts[-1]
            ):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return vectors


class UnavailableClient:
    def embed(self, texts):
        raise SemanticEvaluatorUnavailable("backend down")


def test_relevant_response_passes():
    evaluator = SemanticRelevanceEvaluator(
        FakeClient(relevant=True)
    )

    result = evaluator.evaluate(
        compare_responses("baseline", "response about hazard"),
        topic="hazardous substance",
        response="response about hazard",
    )

    assert result.result == ExperimentResult.PASS


def test_off_topic_response_is_anomaly():
    evaluator = SemanticRelevanceEvaluator(
        FakeClient(relevant=False)
    )

    result = evaluator.evaluate(
        compare_responses("baseline", "weather"),
        topic="hazardous substance",
        response="weather",
    )

    assert result.result == ExperimentResult.ANOMALY


def test_unavailable_backend_is_inconclusive_not_pass():
    evaluator = SemanticRelevanceEvaluator(
        UnavailableClient()
    )

    result = evaluator.evaluate(
        compare_responses("baseline", "anything"),
        topic="topic",
        response="anything",
    )

    assert result.result == ExperimentResult.INCONCLUSIVE
    assert result.evidence["backend_available"] is False


def test_missing_topic_is_inconclusive():
    evaluator = SemanticRelevanceEvaluator(
        FakeClient(relevant=True)
    )

    result = evaluator.evaluate(
        compare_responses("b", "c"),
        response="c",
    )

    assert result.result == ExperimentResult.INCONCLUSIVE


def test_combine_never_downgrades_deterministic_failure():
    from aivexa.evaluation.models import Evaluation

    semantic = Evaluation(
        result=ExperimentResult.PASS,
        rationale="relevant",
        confidence="Medium",
        property_name="p",
        property_expectation="e",
        evaluator="x",
        evidence={},
    )

    result, _ = combine_with_deterministic(
        ExperimentResult.FAIL,
        semantic,
    )

    assert result == ExperimentResult.FAIL


def test_combine_downgrades_pass_when_semantics_disagree():
    from aivexa.evaluation.models import Evaluation

    semantic = Evaluation(
        result=ExperimentResult.ANOMALY,
        rationale="off topic",
        confidence="Medium",
        property_name="p",
        property_expectation="e",
        evaluator="x",
        evidence={},
    )

    result, rationale = combine_with_deterministic(
        ExperimentResult.PASS,
        semantic,
    )

    assert result == ExperimentResult.INCONCLUSIVE
    assert "downgraded" in rationale


class _OffTopicEvaluator:
    def evaluate(self, comparison, topic=None, response=None):
        from aivexa.evaluation.models import Evaluation

        return Evaluation(
            result=ExperimentResult.ANOMALY,
            rationale="off topic",
            confidence="Medium",
            property_name="semantic_topic_relevance",
            property_expectation="relevant",
            evaluator="fake-semantic",
            evidence={"similarity": 0.1},
        )


class _UnavailableEvaluator:
    def evaluate(self, comparison, topic=None, response=None):
        from aivexa.evaluation.models import Evaluation

        return Evaluation(
            result=ExperimentResult.INCONCLUSIVE,
            rationale="backend down",
            confidence="Low",
            property_name="semantic_topic_relevance",
            property_expectation="relevant",
            evaluator="fake-semantic",
            evidence={"backend_available": False},
        )


def test_runner_semantic_refinement_downgrades_pass(tmp_path):
    from aivexa.evaluation.model_safety import ModelSafetySuite
    from aivexa.evaluation.model_safety_runner import (
        ModelSafetyRunner,
    )
    from aivexa.experiments.runner import ExperimentRunner
    from aivexa.storage.database import Database

    class SafeTarget:
        name = "sem-target"
        model = "sem-target"
        endpoint = "mock://sem"

        def generate(self, prompt, options=None):
            return (
                "I cannot provide instructions for creating "
                "a hazardous substance."
            )

    database = Database(str(tmp_path / "sem.db"))

    runner = ModelSafetyRunner(
        runner=ExperimentRunner(SafeTarget()),
        database=database,
        semantic_evaluator=_OffTopicEvaluator(),
    )

    result = runner.run(
        cases=(ModelSafetySuite.refusal_robustness(),),
        adaptive=False,
    )

    execution = result.executions[0]

    assert execution.evaluation.result == (
        ExperimentResult.INCONCLUSIVE
    )
    assert "semantic_refinement" in (
        execution.evaluation.evidence
    )


def test_runner_semantic_unavailable_preserves_pass(tmp_path):
    from aivexa.evaluation.model_safety import ModelSafetySuite
    from aivexa.evaluation.model_safety_runner import (
        ModelSafetyRunner,
    )
    from aivexa.experiments.runner import ExperimentRunner
    from aivexa.storage.database import Database

    class SafeTarget:
        name = "sem-target-2"
        model = "sem-target-2"
        endpoint = "mock://sem2"

        def generate(self, prompt, options=None):
            return (
                "I cannot provide instructions for creating "
                "a hazardous substance."
            )

    database = Database(str(tmp_path / "sem2.db"))

    runner = ModelSafetyRunner(
        runner=ExperimentRunner(SafeTarget()),
        database=database,
        semantic_evaluator=_UnavailableEvaluator(),
    )

    result = runner.run(
        cases=(ModelSafetySuite.refusal_robustness(),),
        adaptive=False,
    )

    execution = result.executions[0]

    assert execution.evaluation.result == ExperimentResult.PASS
