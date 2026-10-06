from dataclasses import dataclass
from typing import Any

from aivexa.evaluation.adaptive import AdaptiveProbe
from aivexa.experiments.models import Experiment


@dataclass(frozen=True)
class AdaptiveExperiment:
    experiment: Experiment
    probe: AdaptiveProbe

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": {
                "experiment_id": self.experiment.experiment_id,
                "target": self.experiment.target,
                "objective": self.experiment.objective,
                "hypothesis": self.experiment.hypothesis,
                "intervention": self.experiment.intervention,
                "context": self.experiment.context,
                "created_at": self.experiment.created_at,
            },
            "probe": self.probe.to_dict(),
        }


class AdaptiveExperimentBuilder:
    """
    Converts an adaptive probe decision into the existing AIVEXA
    Experiment model.

    This does not execute the experiment.
    """

    def build(
        self,
        probe: AdaptiveProbe,
        target: str,
        prompt: str,
        source_experiment_id: str,
    ) -> AdaptiveExperiment:
        experiment = Experiment(
            experiment_id=f"{probe.probe_id}-EXP-001",
            target=target,
            objective=probe.objective,
            hypothesis=probe.hypothesis,
            intervention=probe.intervention_type,
            context={
                "adaptive": True,
                "source_experiment_id": source_experiment_id,
                "probe": probe.to_dict(),
                "prompt": prompt,
            },
        )

        return AdaptiveExperiment(
            experiment=experiment,
            probe=probe,
        )
