from dataclasses import dataclass
from enum import Enum


class ExperimentResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ANOMALY = "ANOMALY"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class Evaluation:
    result: ExperimentResult
    rationale: str
    confidence: str
