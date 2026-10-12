from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


def _hash_content(*parts: str) -> str:
    payload = "\n".join(parts)
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CorpusEntry:
    """
    One piece of research material with provenance.

    An entry represents an advisory, vulnerability record, pattern, or
    report fragment. Provenance fields are mandatory so downstream
    hypotheses can be traced back to the material that informed them.
    """

    entry_id: str
    title: str
    source: str
    summary: str = ""
    url: str = ""
    published_at: str = ""
    retrieved_at: str = ""
    cwes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    severity: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return _hash_content(
            self.entry_id,
            self.title,
            self.summary,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["content_hash"] = self.content_hash
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusEntry":
        return cls(
            entry_id=str(data["entry_id"]),
            title=str(data.get("title", "")),
            source=str(data.get("source", "")),
            summary=str(data.get("summary", "")),
            url=str(data.get("url", "")),
            published_at=str(data.get("published_at", "")),
            retrieved_at=str(data.get("retrieved_at", "")),
            cwes=tuple(data.get("cwes", ())),
            tags=tuple(data.get("tags", ())),
            keywords=tuple(data.get("keywords", ())),
            severity=str(data.get("severity", "")),
            metadata=dict(data.get("metadata", {})),
        )
