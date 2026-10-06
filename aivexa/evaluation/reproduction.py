from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aivexa.evaluation.finding import Finding
from aivexa.evaluation.models import ExperimentResult


class ReproductionStatus(str, Enum):
    PENDING = "PENDING"
    REPRODUCED = "REPRODUCED"
    NOT_REPRODUCED = "NOT_REPRODUCED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class Reproduction:
    reproduction_id: str
    finding_id: str
    source_experiment_id: str
    follow_up_experiment_id: str
    status: ReproductionStatus
    result: ExperimentResult
    confidence: str
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)


class ReproductionBuilder:
    """
    Creates a controlled reproduction record for an existing finding.

    The reproduction is based on a new experiment result; it does
    not modify the original finding or overwrite its evidence.
    """

    def create(
        self,
        reproduction_id: str,
        finding: Finding,
        follow_up_experiment_id: str,
    ) -> Reproduction:
        return Reproduction(
            reproduction_id=reproduction_id,
            finding_id=finding.finding_id,
            source_experiment_id=finding.experiment_id,
            follow_up_experiment_id=follow_up_experiment_id,
            status=ReproductionStatus.PENDING,
            result=ExperimentResult.INCONCLUSIVE,
            confidence="Low",
            rationale=(
                "A controlled follow-up experiment has been "
                "created to reproduce the finding."
            ),
            evidence={
                "source_finding": finding.finding_id,
                "source_assessment": finding.assessment_id,
                "affected_properties": list(
                    finding.affected_properties
                ),
            },
        )

    def resolve(
        self,
        reproduction: Reproduction,
        result: ExperimentResult,
        confidence: str,
        rationale: str,
        evidence: dict[str, Any] | None = None,
    ) -> Reproduction:
        if result == ExperimentResult.FAIL:
            status = ReproductionStatus.REPRODUCED
        elif result == ExperimentResult.PASS:
            status = ReproductionStatus.NOT_REPRODUCED
        else:
            status = ReproductionStatus.INCONCLUSIVE

        merged_evidence = dict(reproduction.evidence)
        merged_evidence["follow_up_result"] = {
            "result": result.value,
            "confidence": confidence,
            "evidence": evidence or {},
        }

        return Reproduction(
            reproduction_id=reproduction.reproduction_id,
            finding_id=reproduction.finding_id,
            source_experiment_id=reproduction.source_experiment_id,
            follow_up_experiment_id=(
                reproduction.follow_up_experiment_id
            ),
            status=status,
            result=result,
            confidence=confidence,
            rationale=rationale,
            evidence=merged_evidence,
        )
