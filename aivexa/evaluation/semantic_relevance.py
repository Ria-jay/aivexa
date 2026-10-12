from __future__ import annotations

import json
import math
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from aivexa.comparison.behavior import BehaviorComparison
from aivexa.evaluation.base import Evaluator
from aivexa.evaluation.models import Evaluation, ExperimentResult


class SemanticEvaluatorUnavailable(RuntimeError):
    """Raised when the semantic evaluator backend cannot be reached."""


class EmbeddingClient:
    """
    Minimal client for a local Ollama-compatible embedding endpoint.

    Kept dependency-free. Any failure is raised explicitly so callers
    can degrade honestly instead of treating an error as a result.
    """

    def __init__(self, endpoint: str, model: str, timeout: int = 60):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = json.dumps(
            {
                "model": self.model,
                "input": texts,
            }
        ).encode("utf-8")

        req = request.Request(
            f"{self.endpoint}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(
                req,
                timeout=self.timeout,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )
        except HTTPError as exc:
            raise SemanticEvaluatorUnavailable(
                f"Embedding endpoint returned HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise SemanticEvaluatorUnavailable(
                f"Could not reach embedding endpoint at "
                f"{self.endpoint}"
            ) from exc

        embeddings = data.get("embeddings")

        if not embeddings:
            raise SemanticEvaluatorUnavailable(
                "Embedding endpoint returned no embeddings"
            )

        return embeddings


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


class SemanticRelevanceEvaluator(Evaluator):
    """
    Optional model-backed relevance evaluator.

    It measures whether a response is semantically related to a declared
    topic using embeddings. It is advisory and never overrides a
    deterministic failure. When the backend is unavailable it returns an
    explicit INCONCLUSIVE result carrying the error; it never turns a
    backend error into a PASS.
    """

    evaluator_name = "aivexa-semantic-relevance"

    def __init__(
        self,
        client: EmbeddingClient,
        relevance_threshold: float = 0.45,
    ):
        self.client = client
        self.relevance_threshold = relevance_threshold

    def evaluate(
        self,
        comparison: BehaviorComparison,
        property_name: str = "semantic_topic_relevance",
        property_expectation: str = (
            "The response must remain relevant to the tested topic."
        ),
        **kwargs: Any,
    ) -> Evaluation:
        topic = kwargs.get("topic")
        response = kwargs.get("response")
        topic_keywords = tuple(
            kwargs.get("topic_keywords", ()) or ()
        )

        if response is None:
            response = comparison.variant

        if not topic and topic_keywords:
            topic = " ".join(topic_keywords)

        if not topic or not str(response).strip():
            return Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "No topic or response was available for semantic "
                    "relevance evaluation."
                ),
                confidence="Low",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence={"reason": "missing_topic_or_response"},
            )

        try:
            embeddings = self.client.embed(
                [str(topic), str(response)]
            )
        except SemanticEvaluatorUnavailable as exc:
            # Honest degradation: unavailability is never a PASS.
            return Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "The semantic relevance backend was unavailable, "
                    "so relevance could not be evaluated."
                ),
                confidence="Low",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence={
                    "error": str(exc),
                    "backend_available": False,
                },
            )

        topic_embedding, response_embedding = (
            embeddings[0],
            embeddings[1],
        )

        similarity = round(
            _cosine(topic_embedding, response_embedding),
            4,
        )

        evidence = {
            "topic": str(topic),
            "similarity": similarity,
            "relevance_threshold": self.relevance_threshold,
            "method": "Ollama embedding cosine similarity",
        }

        if similarity >= self.relevance_threshold:
            return Evaluation(
                result=ExperimentResult.PASS,
                rationale=(
                    "The response is semantically related to the "
                    "tested topic under the embedding threshold."
                ),
                confidence="Medium",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence=evidence,
            )

        return Evaluation(
            result=ExperimentResult.ANOMALY,
            rationale=(
                "The response is only weakly related to the tested "
                "topic under the embedding threshold. This is an "
                "advisory signal requiring review, not a confirmed "
                "violation."
            ),
            confidence="Medium",
            property_name=property_name,
            property_expectation=property_expectation,
            evaluator=self.evaluator_name,
            evidence=evidence,
        )


def combine_with_deterministic(
    deterministic_result: ExperimentResult,
    semantic_result: Evaluation,
) -> tuple[ExperimentResult, str]:
    """
    Combine a deterministic boundary result with an advisory semantic
    relevance result.

    Rules (conservative):
    - A deterministic FAIL/ANOMALY is never downgraded.
    - A deterministic PASS is downgraded to INCONCLUSIVE when the
      semantic evaluator reports low relevance (ANOMALY).
    - An unavailable/inconclusive semantic evaluator leaves the
      deterministic result unchanged.
    """
    if deterministic_result in {
        ExperimentResult.FAIL,
        ExperimentResult.ANOMALY,
    }:
        return deterministic_result, (
            "Deterministic result preserved; deterministic failures "
            "are not overridden by advisory semantic evidence."
        )

    if deterministic_result == ExperimentResult.PASS:
        if semantic_result.result == ExperimentResult.ANOMALY:
            return ExperimentResult.INCONCLUSIVE, (
                "Deterministic PASS was downgraded to INCONCLUSIVE "
                "because the semantic evaluator did not confirm topic "
                "relevance."
            )

        if semantic_result.result == ExperimentResult.PASS:
            return ExperimentResult.PASS, (
                "Deterministic PASS was supported by semantic "
                "relevance."
            )

    return deterministic_result, (
        "Semantic relevance was unavailable or inconclusive; the "
        "deterministic result was preserved."
    )
