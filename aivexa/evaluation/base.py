from abc import ABC, abstractmethod
from typing import Any

from aivexa.comparison.behavior import BehaviorComparison
from aivexa.evaluation.models import Evaluation


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        comparison: BehaviorComparison,
        property_name: str,
        property_expectation: str,
        **kwargs: Any,
    ) -> Evaluation:
        raise NotImplementedError
