from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Experiment:
    experiment_id: str
    target: str
    objective: str
    hypothesis: str
    intervention: str
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
