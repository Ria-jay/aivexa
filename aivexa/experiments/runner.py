from aivexa.evidence.models import Evidence
from aivexa.targets.ollama import OllamaTarget


class ExperimentRunner:
    def __init__(self, target: OllamaTarget):
        self.target = target

    def run(
        self,
        experiment_id: str,
        prompt: str,
    ) -> Evidence:
        response = self.target.generate(prompt)

        return Evidence(
            experiment_id=experiment_id,
            input_data=prompt,
            output_data=response,
            observations=[],
            metadata={
                "target_type": "ollama",
                "model": self.target.model,
                "endpoint": self.target.endpoint,
            },
        )
