import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TargetConfig:
    name: str
    target_type: str
    endpoint: str
    model: str = ""
    api_key_env: str = ""
    timeout: int = 120
    capabilities: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetConfig":
        return cls(
            name=data["name"],
            target_type=data["target_type"],
            endpoint=data.get("endpoint", ""),
            model=data.get("model", ""),
            api_key_env=data.get("api_key_env", ""),
            timeout=int(data.get("timeout", 120)),
            capabilities=dict(data.get("capabilities", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def resolve_api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


class TargetRegistry:
    def __init__(self, path: str = "aivexa-targets.json"):
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"targets": {}}

        return json.loads(self.path.read_text())

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(
            json.dumps(data, indent=2, sort_keys=True)
        )

    def save(self, config: TargetConfig) -> None:
        data = self._load()
        data.setdefault("targets", {})
        data["targets"][config.name] = config.to_dict()
        self._save(data)

    def get(self, name: str) -> TargetConfig:
        data = self._load()
        try:
            return TargetConfig.from_dict(
                data["targets"][name]
            )
        except KeyError as exc:
            raise KeyError(
                f"Target '{name}' is not configured."
            ) from exc

    def list(self) -> list[TargetConfig]:
        data = self._load()
        return [
            TargetConfig.from_dict(value)
            for value in data.get("targets", {}).values()
        ]

    def delete(self, name: str) -> None:
        data = self._load()
        targets = data.get("targets", {})

        if name not in targets:
            raise KeyError(
                f"Target '{name}' is not configured."
            )

        del targets[name]
        self._save(data)
