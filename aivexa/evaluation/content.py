from dataclasses import dataclass


@dataclass(frozen=True)
class ContentEvaluation:
    preserved: bool
    matched_concepts: list[str]
    missing_concepts: list[str]
    unexpected_concepts: list[str]
    rationale: str


def evaluate_content_preservation(
    baseline: str,
    variant: str,
    required_concepts: list[str],
) -> ContentEvaluation:
    baseline_words = set(baseline.lower().split())
    variant_words = set(variant.lower().split())

    matched = [
        concept
        for concept in required_concepts
        if concept.lower() in baseline_words
        and concept.lower() in variant_words
    ]

    missing = [
        concept
        for concept in required_concepts
        if concept.lower() in baseline_words
        and concept.lower() not in variant_words
    ]

    preserved = not missing

    return ContentEvaluation(
        preserved=preserved,
        matched_concepts=matched,
        missing_concepts=missing,
        unexpected_concepts=[],
        rationale=(
            "Deterministic lexical comparison only. "
            "This evaluator must not be interpreted as semantic judgment."
        ),
    )
