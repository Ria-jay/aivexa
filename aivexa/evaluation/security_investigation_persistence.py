import json

from aivexa.evaluation.security_investigation import SecurityInvestigation
from aivexa.storage.database import Database


class SecurityInvestigationPersistence:
    """
    Persists a Phase 4 security investigation as an AIVEXA experiment.

    The source experiment, assessment, affected properties, decision,
    hypothesis, and suggested follow-up work remain explicitly traceable.
    """

    def __init__(self, database: Database):
        self.database = database

    def save(self, investigation: SecurityInvestigation) -> None:
        experiment = investigation.experiment
        context = dict(experiment.context)

        context["investigation_lineage"] = {
            "source_experiment_id": investigation.source_experiment_id,
            "follow_up_experiment_id": experiment.experiment_id,
            "follow_up_action": investigation.follow_up.action,
        }

        self.database.save_experiment(
            {
                "experiment_id": experiment.experiment_id,
                "target": experiment.target,
                "objective": experiment.objective,
                "hypothesis": experiment.hypothesis,
                "intervention": experiment.intervention,
                "context": context,
                "created_at": experiment.created_at,
                "input_data": json.dumps(
                    {
                        "source_experiment_id":
                            investigation.source_experiment_id,
                        "suggested_experiments":
                            list(
                                investigation.follow_up
                                .suggested_experiments
                            ),
                    }
                ),
                "output_data": "",
                "result": "INCONCLUSIVE",
                "rationale": investigation.follow_up.reason,
                "confidence": "Low",
                "observations": [],
                "property_name": (
                    investigation.follow_up.affected_properties[0]
                    if investigation.follow_up.affected_properties
                    else "security_investigation"
                ),
                "property_expectation": (
                    investigation.follow_up.hypothesis
                ),
                "comparison_group": (
                    investigation.source_experiment_id
                ),
                "evaluator": "aivexa-security-investigation",
            }
        )
