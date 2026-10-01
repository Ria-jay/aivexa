from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExperimentResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ANOMALY = "ANOMALY"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Evaluation:
    result: ExperimentResult
    rationale: str
    confidence: str
    property_name: str
    property_expectation: str
    evaluator: str
    evidence: dict[str, Any]
