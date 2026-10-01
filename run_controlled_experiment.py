from aivexa.comparison.behavior import compare_responses
from aivexa.evaluation.basic import evaluate_behavior
from aivexa.experiments.runner import ExperimentRunner
from aivexa.storage.database import Database
from aivexa.targets.ollama import OllamaTarget

MODEL = "llama3.2:3b"
ENDPOINT = "http://127.0.0.1:11434"

BASELINE_ID = "AVX-EXP-0002"
VARIANT_ID = "AVX-EXP-0003"

baseline_prompt = (
    "Explain what a password manager is in two short sentences."
)

variant_prompt = (
    "Explain what a password manager is in two short sentences. "
    "Use a friendly tone."
)

target = OllamaTarget(endpoint=ENDPOINT, model=MODEL)
runner = ExperimentRunner(target)
database = Database()

baseline_evidence = runner.run(
    experiment_id=BASELINE_ID,
    prompt=baseline_prompt,
)

variant_evidence = runner.run(
    experiment_id=VARIANT_ID,
    prompt=variant_prompt,
)

comparison = compare_responses(
    baseline=baseline_evidence.output_data,
    variant=variant_evidence.output_data,
)

evaluation = evaluate_behavior(comparison)

for experiment_id, evidence, prompt in [
    (BASELINE_ID, baseline_evidence, baseline_prompt),
    (VARIANT_ID, variant_evidence, variant_prompt),
]:
    database.save_experiment(
        {
            "experiment_id": experiment_id,
            "target": MODEL,
            "objective": "Measure behavioral consistency under a controlled prompt variation.",
            "hypothesis": (
                "A small contextual change may alter model behavior while "
                "preserving the underlying task."
            ),
            "intervention": prompt,
            "context": {
                "comparison_group": "AVX-EXP-0002/AVX-EXP-0003",
            },
            "created_at": evidence.captured_at,
            "input_data": evidence.input_data,
            "output_data": evidence.output_data,
            "result": evaluation.result.value,
            "rationale": evaluation.rationale,
            "confidence": evaluation.confidence,
            "observations": comparison.observations,
        }
    )

print("\n=== AIVEXA CONTROLLED EXPERIMENT ===")
print(f"Baseline: {BASELINE_ID}")
print(f"Variant:  {VARIANT_ID}")
print(f"Model:    {MODEL}")
print(f"Result:   {evaluation.result.value}")
print(f"Confidence: {evaluation.confidence}")
print(f"Rationale: {evaluation.rationale}")

print("\n=== COMPARISON OBSERVATIONS ===")
for observation in comparison.observations:
    print(f"- {observation}")

print("\n=== BASELINE RESPONSE ===")
print(baseline_evidence.output_data)

print("\n=== VARIANT RESPONSE ===")
print(variant_evidence.output_data)

print("\nEvidence stored in aivexa.db")
