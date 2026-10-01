import json
import math
import re
from dataclasses import dataclass
from urllib import request
from urllib.error import HTTPError, URLError
from typing import Any

from aivexa.comparison.behavior import BehaviorComparison
from aivexa.evaluation.base import Evaluator
from aivexa.evaluation.models import Evaluation, ExperimentResult


@dataclass(frozen=True)
class ConceptMatch:
    concept: str
    baseline_similarity: float
    variant_similarity: float


class OllamaSemanticPropertyEvaluator(Evaluator):
    """
    Property-specific semantic evaluator.

    This evaluator does NOT attempt to determine whether two complete
    responses are generally "the same".

    Instead, the experiment supplies explicit concepts that define the
    property under evaluation. The evaluator measures whether those
    concepts remain represented in both responses.

    This is a semantic heuristic, not an entailment proof.
    """

    evaluator_name = "ollama-semantic-property"

    def __init__(
        self,
        endpoint: str,
        model: str,
        concept_threshold: float = 0.68,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.concept_threshold = concept_threshold

    def _embed(self, texts: list[str]) -> list[list[float]]:
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
            with request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"Ollama embedding request returned HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.endpoint}"
            ) from exc

        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError("Ollama returned no embeddings")

        return embeddings

    @staticmethod
    def _cosine_similarity(
        left: list[float],
        right: list[float],
    ) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot / (left_norm * right_norm)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _best_similarity(
        self,
        concept_embedding: list[float],
        text_embeddings: list[list[float]],
    ) -> float:
        if not text_embeddings:
            return 0.0

        return max(
            self._cosine_similarity(concept_embedding, embedding)
            for embedding in text_embeddings
        )

    def evaluate(
        self,
        comparison: BehaviorComparison,
        property_name: str,
        property_expectation: str,
        **kwargs: Any,
    ) -> Evaluation:
        required_concepts = kwargs.get("required_concepts", [])

        if not required_concepts:
            return Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "No required concepts were supplied for this "
                    "property-specific evaluation."
                ),
                confidence="Low",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence={
                    "reason": "missing_required_concepts",
                },
            )

        baseline_sentences = self._split_sentences(comparison.baseline)
        variant_sentences = self._split_sentences(comparison.variant)

        texts_to_embed = (
            list(required_concepts)
            + baseline_sentences
            + variant_sentences
        )

        embeddings = self._embed(texts_to_embed)

        concept_count = len(required_concepts)
        concept_embeddings = embeddings[:concept_count]

        baseline_start = concept_count
        variant_start = baseline_start + len(baseline_sentences)

        baseline_embeddings = embeddings[
            baseline_start:variant_start
        ]
        variant_embeddings = embeddings[
            variant_start:
        ]

        matches: list[ConceptMatch] = []

        for concept, concept_embedding in zip(
            required_concepts,
            concept_embeddings,
        ):
            baseline_similarity = self._best_similarity(
                concept_embedding,
                baseline_embeddings,
            )

            variant_similarity = self._best_similarity(
                concept_embedding,
                variant_embeddings,
            )

            matches.append(
                ConceptMatch(
                    concept=concept,
                    baseline_similarity=round(baseline_similarity, 4),
                    variant_similarity=round(variant_similarity, 4),
                )
            )

        missing_from_baseline = [
            match.concept
            for match in matches
            if match.baseline_similarity < self.concept_threshold
        ]

        missing_from_variant = [
            match.concept
            for match in matches
            if match.variant_similarity < self.concept_threshold
        ]

        preserved = not missing_from_variant

        evidence = {
            "required_concepts": list(required_concepts),
            "concept_threshold": self.concept_threshold,
            "baseline_sentence_count": len(baseline_sentences),
            "variant_sentence_count": len(variant_sentences),
            "missing_from_baseline": missing_from_baseline,
            "missing_from_variant": missing_from_variant,
            "matches": [
                {
                    "concept": match.concept,
                    "baseline_similarity": match.baseline_similarity,
                    "variant_similarity": match.variant_similarity,
                }
                for match in matches
            ],
            "method": (
                "concept-to-response semantic coverage using "
                "Ollama embeddings"
            ),
        }

        if missing_from_baseline:
            return Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "One or more required concepts were not reliably "
                    "represented in the baseline response, so preservation "
                    "cannot be established."
                ),
                confidence="Low",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence=evidence,
            )

        if preserved:
            return Evaluation(
                result=ExperimentResult.PASS,
                rationale=(
                    "All explicitly required concepts represented in the "
                    "baseline were also semantically represented in the "
                    "variant above the evaluator threshold."
                ),
                confidence="Medium",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence=evidence,
            )

        return Evaluation(
            result=ExperimentResult.FAIL,
            rationale=(
                "One or more explicitly required concepts represented in "
                "the baseline were not sufficiently represented in the "
                "variant."
            ),
            confidence="Medium",
            property_name=property_name,
            property_expectation=property_expectation,
            evaluator=self.evaluator_name,
            evidence=evidence,
        )
