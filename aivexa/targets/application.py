from abc import ABC, abstractmethod
from typing import Any

from aivexa.evidence.system import SystemInteraction


class AIApplicationTarget(ABC):
    """
    Base interface for an AI-powered application target.

    A target represents an authorized system under evaluation.
    The target is responsible for executing an interaction and returning
    observable system behavior.
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
    ):
        self.name = name
        self.endpoint = endpoint

    @abstractmethod
    def interact(
        self,
        experiment_id: str,
        user_input: str,
        context: dict[str, Any] | None = None,
        authorization_context: dict[str, Any] | None = None,
    ) -> SystemInteraction:
        raise NotImplementedError
