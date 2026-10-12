from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from aivexa.knowledge.models import CorpusEntry


class CorpusFormatError(ValueError):
    """Raised when a corpus source cannot be parsed."""


@dataclass(frozen=True)
class IngestResult:
    added: int
    updated: int
    unchanged: int
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "errors": list(self.errors),
        }


class KnowledgeCorpus:
    """
    File-backed research corpus with provenance, deduplication, and
    update tracking.

    The corpus is deliberately independent of the experiment database.
    It stores advisory/pattern/report material that can inform
    hypotheses and follow-up experiments, with explicit source
    attribution and timestamps so decisions remain traceable.

    This class does not fetch live feeds. Live refresh is the
    responsibility of a ``CorpusSource``; the corpus only records what
    was actually ingested.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, CorpusEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._entries = {}
            return

        entries: dict[str, CorpusEntry] = {}

        for line_number, line in enumerate(
            self.path.read_text(
                encoding="utf-8"
            ).splitlines(),
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusFormatError(
                    f"{self.path}:{line_number}: invalid JSON: {exc}"
                ) from exc

            try:
                entry = CorpusEntry.from_dict(data)
            except (KeyError, TypeError) as exc:
                raise CorpusFormatError(
                    f"{self.path}:{line_number}: invalid entry: {exc}"
                ) from exc

            entries[entry.entry_id] = entry

        self._entries = entries

    def save(self) -> Path:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        lines = [
            json.dumps(
                entry.to_dict(),
                sort_keys=True,
                ensure_ascii=False,
            )
            for entry in sorted(
                self._entries.values(),
                key=lambda item: item.entry_id,
            )
        ]

        self.path.write_text(
            "\n".join(lines)
            + ("\n" if lines else ""),
            encoding="utf-8",
        )

        return self.path

    def add(
        self,
        entry: CorpusEntry,
        retrieved_at: str | None = None,
    ) -> bool:
        """
        Add or update one entry. Returns True if the stored content
        changed, False if it was unchanged.
        """
        normalized = self._normalize(
            entry,
            retrieved_at=retrieved_at,
        )

        existing = self._entries.get(entry.entry_id)

        if (
            existing is not None
            and existing.content_hash == normalized.content_hash
        ):
            return False

        self._entries[entry.entry_id] = normalized
        return True

    def ingest(
        self,
        entries: Iterable[CorpusEntry],
    ) -> IngestResult:
        added = updated = unchanged = 0

        for entry in entries:
            existed = entry.entry_id in self._entries
            changed = self.add(entry)

            if not existed and changed:
                added += 1
            elif changed:
                updated += 1
            else:
                unchanged += 1

        return IngestResult(
            added=added,
            updated=updated,
            unchanged=unchanged,
        )

    @staticmethod
    def _normalize(
        entry: CorpusEntry,
        retrieved_at: str | None = None,
    ) -> CorpusEntry:
        stamp = retrieved_at or entry.retrieved_at or (
            datetime.now(timezone.utc).isoformat()
        )

        return CorpusEntry(
            entry_id=entry.entry_id,
            title=entry.title,
            source=entry.source,
            summary=entry.summary,
            url=entry.url,
            published_at=entry.published_at,
            retrieved_at=stamp,
            cwes=tuple(entry.cwes),
            tags=tuple(entry.tags),
            keywords=tuple(entry.keywords),
            severity=entry.severity,
            metadata=dict(entry.metadata),
        )

    def get(self, entry_id: str) -> CorpusEntry | None:
        return self._entries.get(entry_id)

    def all(self) -> list[CorpusEntry]:
        return sorted(
            self._entries.values(),
            key=lambda item: item.entry_id,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def search(
        self,
        query: str | None = None,
        cwe: str | None = None,
        tag: str | None = None,
        limit: int = 10,
    ) -> list[CorpusEntry]:
        terms = [
            term
            for term in (query or "").lower().split()
            if term
        ]

        results: list[tuple[int, CorpusEntry]] = []

        for entry in self._entries.values():
            if cwe and cwe.upper() not in {
                value.upper() for value in entry.cwes
            }:
                continue

            if tag and tag.lower() not in {
                value.lower() for value in entry.tags
            }:
                continue

            haystack = " ".join(
                [
                    entry.title,
                    entry.summary,
                    " ".join(entry.keywords),
                    " ".join(entry.tags),
                    " ".join(entry.cwes),
                ]
            ).lower()

            score = sum(
                1 for term in terms if term in haystack
            )

            if terms and score == 0:
                continue

            results.append((score, entry))

        results.sort(
            key=lambda item: (-item[0], item[1].entry_id)
        )

        return [
            entry for _, entry in results[:limit]
        ]

    def is_stale(
        self,
        entry: CorpusEntry,
        max_age_days: int,
        now: datetime | None = None,
    ) -> bool:
        if not entry.retrieved_at:
            return True

        try:
            retrieved = datetime.fromisoformat(
                entry.retrieved_at
            )
        except ValueError:
            return True

        if retrieved.tzinfo is None:
            retrieved = retrieved.replace(
                tzinfo=timezone.utc
            )

        reference = now or datetime.now(timezone.utc)

        age = reference - retrieved

        return age.days > max_age_days
