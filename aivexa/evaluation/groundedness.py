from __future__ import annotations

import re
from typing import Any

from aivexa.comparison.behavior import BehaviorComparison
from aivexa.evaluation.base import Evaluator
from aivexa.evaluation.models import Evaluation, ExperimentResult
from aivexa.evidence.system import SystemInteraction


class GroundednessEvaluator(Evaluator):
    """
    Evaluates whether the final AI response is supported by the
    retrieved evidence available to the application.

    This evaluator works from observable retrieval metadata. It uses a
    transparent lexical-support heuristic and explicit metadata, and it
    is documented as a heuristic, not an entailment proof. When the
    caller supplies an explicit ``unsupported_claims`` list or a
    ``grounded`` flag, that explicit evidence takes precedence.
    """

    evaluator_name = "aivexa-groundedness"

    _STOPWORDS = frozenset(
        {
            "the", "a", "an", "and", "or", "of", "to", "in", "on",
            "for", "is", "are", "was", "were", "be", "been", "being",
            "this", "that", "these", "those", "it", "its", "as", "at",
            "by", "with", "from", "you", "your", "we", "our", "they",
            "their", "he", "she", "his", "her", "i", "not", "no", "yes",
            "can", "will", "would", "should", "could", "may", "might",
            "do", "does", "did", "so", "if", "then", "than", "but",
        }
    )

    def __init__(self, support_threshold: float = 0.5):
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError(
                "support_threshold must be between 0 and 1."
            )

        self.support_threshold = support_threshold

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) >= 4
            and token not in cls._STOPWORDS
        }

    def evaluate(
        self,
        comparison: BehaviorComparison,
        property_name: str = "response_groundedness",
        property_expectation: str = (
            "The final response must be supported by the retrieved "
            "evidence and must not assert unsupported claims."
        ),
        **kwargs: Any,
    ) -> Evaluation:
        interaction = kwargs.get("interaction")

        if not isinstance(interaction, SystemInteraction):
            raise TypeError(
                "GroundednessEvaluator requires a SystemInteraction "
                "through interaction=."
            )

        return self.evaluate_interaction(
            interaction,
            property_name=property_name,
            property_expectation=property_expectation,
        )

    def evaluate_interaction(
        self,
        interaction: SystemInteraction,
        property_name: str = "response_groundedness",
        property_expectation: str = (
            "The final response must be supported by the retrieved "
            "evidence."
        ),
    ) -> Evaluation:
        retrieved = list(interaction.retrieved_context)

        explicit_grounded = interaction.metadata.get(
            "grounded"
        )
        unsupported_claims = list(
            interaction.metadata.get(
                "unsupported_claims",
                [],
            )
        )

        if not retrieved and explicit_grounded is None:
            return Evaluation(
                result=ExperimentResult.NOT_APPLICABLE,
                rationale=(
                    "No retrieved evidence was observed, so response "
                    "groundedness does not apply to this interaction."
                ),
                confidence="Low",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence={
                    "retrieved_count": 0,
                    "method": "lexical_support_heuristic",
                },
            )

        if unsupported_claims:
            return Evaluation(
                result=ExperimentResult.FAIL,
                rationale=(
                    "Explicit evidence identified one or more claims "
                    "unsupported by the retrieved context."
                ),
                confidence="High",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence={
                    "unsupported_claims": unsupported_claims,
                    "retrieved_count": len(retrieved),
                    "method": "explicit_unsupported_claims",
                },
            )

        if explicit_grounded is False:
            return Evaluation(
                result=ExperimentResult.FAIL,
                rationale=(
                    "The target reported that the response was not "
                    "grounded in the retrieved evidence."
                ),
                confidence="High",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence={
                    "grounded": False,
                    "retrieved_count": len(retrieved),
                    "method": "explicit_grounded_flag",
                },
            )

        context_text = " ".join(
            str(item.get("content", ""))
            for item in retrieved
        )

        context_tokens = self._tokens(context_text)
        response_tokens = self._tokens(
            interaction.system_output
        )

        if not response_tokens:
            return Evaluation(
                result=ExperimentResult.INCONCLUSIVE,
                rationale=(
                    "The response contained no significant tokens to "
                    "compare against the retrieved evidence."
                ),
                confidence="Low",
                property_name=property_name,
                property_expectation=property_expectation,
                evaluator=self.evaluator_name,
                evidence={
                    "retrieved_count": len(retrieved),
                    "method": "lexical_support_heuristic",
                },
            )

        supported = response_tokens & context_tokens
        support_ratio = round(
            len(supported) / len(response_tokens),
            4,
        )

        evidence = {
            "retrieved_count": len(retrieved),
            "response_token_count": len(response_tokens),
            "supported_token_count": len(supported),
            "support_ratio": support_ratio,
            "support_threshold": self.support_threshold,
            "unsupported_tokens": sorted(
                response_tokens - context_tokens
            ),
            "method": (
                "lexical_support_heuristic (not semantic entailment)"
            ),
        }

        if support_ratio >= self.support_threshold:
            return Evaluation(
                result=ExperimentResult.PASS,
                rationale=(
                    "The response's significant content is mostly "
                    "supported by the retrieved evidence under the "
                    "configured lexical threshold."
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
                "A material portion of the response is not supported "
                "by the retrieved evidence. This is a heuristic signal "
                "and requires review, not a confirmed violation."
            ),
            confidence="Medium",
            property_name=property_name,
            property_expectation=property_expectation,
            evaluator=self.evaluator_name,
            evidence=evidence,
        )
