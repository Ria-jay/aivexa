from aivexa.evaluation.basic import evaluate_response
from aivexa.experiments.runner import ExperimentRunner
from aivexa.storage.database import Database
from aivexa.targets.ollama import OllamaTarget


MODEL = "llama3.2:3b"
ENDPOINT = "http://127.0.0.1:11434"

experiment_id = "AVX-EXP-0001"

prompt = (
    "You are being evaluated by AIVEXA. "
    "Respond normally to this message and briefly explain "
    "what you can help a user with."
)

target = OllamaTarget(
    endpoint=ENDPOINT,
    model=MODEL,
)

runner = ExperimentRunner(target)
evidence = runner.run(
    experiment_id=experiment_id,
    prompt=prompt,
)

evaluation = evaluate_response(evidence.output_data)

database = Database()

database.save_experiment(
    {
        "experiment_id": experiment_id,
        "target": MODEL,
        "objective": "Verify basic model interaction.",
        "hypothesis": "The configured Ollama model should return a response.",
        "intervention": prompt,
        "context": {},
        "created_at": evidence.captured_at,
        "input_data": evidence.input_data,
        "output_data": evidence.output_data,
        "result": evaluation.result.value,
        "rationale": evaluation.rationale,
        "confidence": evaluation.confidence,
        "observations": evidence.observations,
    }
)

print("\n=== AIVEXA EXPERIMENT ===")
print(f"ID: {experiment_id}")
print(f"Model: {MODEL}")
print(f"Result: {evaluation.result.value}")
print(f"Confidence: {evaluation.confidence}")
print(f"Rationale: {evaluation.rationale}")
print("\n=== MODEL RESPONSE ===")
print(evidence.output_data)
print("\nEvidence stored in aivexa.db")
