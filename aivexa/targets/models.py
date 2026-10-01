from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str
    target_type: str
    endpoint: str
    model: str
