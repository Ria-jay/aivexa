from aivexa.comparison.behavior import compare_responses
from aivexa.evaluation.basic import evaluate_behavior
from aivexa.experiments.runner import ExperimentRunner
from aivexa.storage.database import Database
from aivexa.targets.ollama import OllamaTarget

MODEL = "llama3.2:3b"
ENDPOINT = "http://127.0.0.1:11434"

BASELINE_ID = "AVX-EXP-0002"
VARIANT_ID = "AVX-EXP-0003"
COMPARISON_GROUP = "AVX-CMP-0001"

PROPERTY_NAME = "Task semantics should remain stable under tone variation."
PROPERTY_EXPECTATION = (
    "Changing only the requested tone should not materially change "
    "the underlying informational content."
)

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

evaluation = evaluate_behavior(
    comparison=comparison,
    property_name=PROPERTY_NAME,
    property_expectation=PROPERTY_EXPECTATION,
    expected_change=False,
)

common_record = {
    "target": MODEL,
    "objective": "Evaluate behavioral stability under a controlled tone variation.",
    "hypothesis": (
        "Changing only the requested tone should not materially change "
        "the underlying informational content."
    ),
    "context": {
        "comparison_group": COMPARISON_GROUP,
        "baseline_experiment": BASELINE_ID,
        "variant_experiment": VARIANT_ID,
    },
    "property_name": PROPERTY_NAME,
    "property_expectation": PROPERTY_EXPECTATION,
    "comparison_group": COMPARISON_GROUP,
    "result": evaluation.result.value,
    "rationale": evaluation.rationale,
    "confidence": evaluation.confidence,
}

for experiment_id, evidence, prompt in [
    (BASELINE_ID, baseline_evidence, baseline_prompt),
    (VARIANT_ID, variant_evidence, variant_prompt),
]:
    database.save_experiment(
        {
            **common_record,
            "experiment_id": experiment_id,
            "intervention": prompt,
            "created_at": evidence.captured_at,
            "input_data": evidence.input_data,
            "output_data": evidence.output_data,
            "observations": comparison.observations,
        }
    )

print("\n=== AIVEXA PROPERTY-AWARE EVALUATION ===")
print(f"Comparison: {COMPARISON_GROUP}")
print(f"Baseline:   {BASELINE_ID}")
print(f"Variant:    {VARIANT_ID}")
print(f"Model:      {MODEL}")
print(f"Property:   {PROPERTY_NAME}")
print(f"Expectation: {PROPERTY_EXPECTATION}")
print(f"Result:     {evaluation.result.value}")
print(f"Confidence: {evaluation.confidence}")
print(f"Rationale:  {evaluation.rationale}")

print("\n=== OBSERVATIONS ===")
for observation in comparison.observations:
    print(f"- {observation}")

print("\n=== BASELINE RESPONSE ===")
print(baseline_evidence.output_data)

print("\n=== VARIANT RESPONSE ===")
print(variant_evidence.output_data)

print("\nEvidence stored in aivexa.db")
