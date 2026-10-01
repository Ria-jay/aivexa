from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Evidence:
    experiment_id: str
    input_data: str
    output_data: str
    observations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
