from aivexa.comparison.behavior import BehaviorComparison
from aivexa.evaluation.semantic import OllamaSemanticPropertyEvaluator
from aivexa.experiments.runner import ExperimentRunner
from aivexa.storage.database import Database
from aivexa.targets.ollama import OllamaTarget


ENDPOINT = "http://127.0.0.1:11434"
MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

EXPERIMENT_ID = "AVX-EXP-0001"
COMPARISON_GROUP = "AVX-CMP-0001"

PROPERTY_NAME = "content_preservation_under_tone_variation"

PROPERTY_EXPECTATION = (
    "Changing only the requested tone should preserve the explicitly "
    "defined informational concepts."
)

BASELINE_PROMPT = (
    "Explain what a password manager is in two short sentences."
)

VARIANT_PROMPT = (
    "Explain what a password manager is in two short sentences. "
    "Use a friendly tone."
)

REQUIRED_CONCEPTS = [
    "securely stores login passwords",
    "generates strong unique passwords",
    "helps manage credentials for online accounts",
    "protects stored credentials",
]


def main() -> None:
    target = OllamaTarget(
        endpoint=ENDPOINT,
        model=MODEL,
    )

    runner = ExperimentRunner(target)

    baseline_evidence = runner.run(
        experiment_id=f"{EXPERIMENT_ID}-BASELINE",
        prompt=BASELINE_PROMPT,
    )

    variant_evidence = runner.run(
        experiment_id=f"{EXPERIMENT_ID}-VARIANT",
        prompt=VARIANT_PROMPT,
    )

    changed = (
        baseline_evidence.output_data.strip()
        != variant_evidence.output_data.strip()
    )

    observations = [
        "Baseline and variant were generated independently.",
        f"Response text changed: {'yes' if changed else 'no'}.",
        f"Baseline length: {len(baseline_evidence.output_data)} characters.",
        f"Variant length: {len(variant_evidence.output_data)} characters.",
    ]

    comparison = BehaviorComparison(
        baseline=baseline_evidence.output_data,
        variant=variant_evidence.output_data,
        changed=changed,
        observations=observations,
    )

    evaluator = OllamaSemanticPropertyEvaluator(
        endpoint=ENDPOINT,
        model=EMBEDDING_MODEL,
    )

    evaluation = evaluator.evaluate(
        comparison=comparison,
        property_name=PROPERTY_NAME,
        property_expectation=PROPERTY_EXPECTATION,
        required_concepts=REQUIRED_CONCEPTS,
    )

    experiment = {
        "experiment_id": EXPERIMENT_ID,
        "target": target.model,
        "objective": (
            "Evaluate content preservation under tone variation."
        ),
        "hypothesis": (
            "Changing tone should not remove the explicitly defined "
            "informational concepts."
        ),
        "intervention": VARIANT_PROMPT,
        "context": {
            "comparison_group": COMPARISON_GROUP,
            "baseline_prompt": BASELINE_PROMPT,
            "variant_prompt": VARIANT_PROMPT,
            "required_concepts": REQUIRED_CONCEPTS,
            "baseline_output": baseline_evidence.output_data,
            "variant_output": variant_evidence.output_data,
        },
        "created_at": evaluation.evidence.get(
            "created_at",
            baseline_evidence.captured_at,
        ),
        "input_data": VARIANT_PROMPT,
        "output_data": variant_evidence.output_data,
        "result": evaluation.result.value,
        "rationale": evaluation.rationale,
        "confidence": evaluation.confidence,
        "observations": observations,
        "property_name": evaluation.property_name,
        "property_expectation": evaluation.property_expectation,
        "comparison_group": COMPARISON_GROUP,
        "evaluator": evaluation.evaluator,
    }

    database = Database()
    database.save_experiment(experiment)

    print()
    print("RESULT:", evaluation.result.value)
    print("CONFIDENCE:", evaluation.confidence)
    print("EVALUATOR:", evaluation.evaluator)
    print("PROPERTY:", evaluation.property_name)
    print("RATIONALE:", evaluation.rationale)

    print()
    print("REQUIRED CONCEPTS:")

    for concept in REQUIRED_CONCEPTS:
        print("-", concept)

    print()
    print("EVIDENCE:")

    for match in evaluation.evidence["matches"]:
        print(
            f"- {match['concept']}"
            f" | baseline={match['baseline_similarity']}"
            f" | variant={match['variant_similarity']}"
        )

    print()
    print("BASELINE:")
    print(baseline_evidence.output_data)

    print()
    print("VARIANT:")
    print(variant_evidence.output_data)


if __name__ == "__main__":
    main()
