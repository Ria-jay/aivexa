from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from aivexa.knowledge.corpus import KnowledgeCorpus
from aivexa.knowledge.models import CorpusEntry


@dataclass(frozen=True)
class RefreshReport:
    """
    Outcome of one corpus refresh attempt.

    ``success`` is True only when at least one page/entry was retrieved
    without error. A partial or failed refresh is reported explicitly;
    it is never presented as a successful, complete refresh.
    """

    source: str
    pages: int
    entries_fetched: int
    added: int
    updated: int
    unchanged: int
    skipped: int
    next_start_index: int
    total_results: int
    exhausted: bool
    success: bool
    started_at: str
    finished_at: str
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def refresh_corpus(
    corpus: KnowledgeCorpus,
    source: Any,
    *,
    max_pages: int = 1,
    start_index: int = 0,
    results_per_page: int = 100,
    last_mod_start: str | None = None,
    last_mod_end: str | None = None,
) -> RefreshReport:
    """
    Fetch entries from a source and ingest them into a corpus.

    Supports both paginated sources exposing ``fetch_all`` (e.g. the NVD
    adapter) and single-shot sources exposing ``fetch`` (local JSONL and
    generic HTTP sources).
    """
    source_name = getattr(source, "name", type(source).__name__)

    started_at = datetime.now(timezone.utc).isoformat()
    errors: list[str] = []
    entries: list[CorpusEntry] = []
    pages = 0
    next_start_index = start_index
    total_results = 0
    exhausted = True
    metadata: dict[str, Any] = {}

    if hasattr(source, "fetch_all"):
        result = source.fetch_all(
            max_pages=max_pages,
            start_index=start_index,
            results_per_page=results_per_page,
            last_mod_start=last_mod_start,
            last_mod_end=last_mod_end,
        )

        entries = list(result.entries)
        pages = result.pages
        next_start_index = result.next_start_index
        total_results = result.total_results
        exhausted = result.exhausted
        errors = list(result.errors)
    else:
        try:
            entries = list(source.fetch())
            pages = 1
            next_start_index = start_index + len(entries)
            total_results = len(entries)
            exhausted = True
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            pages = 0
            next_start_index = start_index
            total_results = 0
            exhausted = False

    ingest_result = corpus.ingest(entries)

    if entries:
        corpus.save()

    finished_at = datetime.now(timezone.utc).isoformat()

    success = bool(entries) and not errors

    if errors and entries:
        metadata["partial"] = True

    return RefreshReport(
        source=source_name,
        pages=pages,
        entries_fetched=len(entries),
        added=ingest_result.added,
        updated=ingest_result.updated,
        unchanged=ingest_result.unchanged,
        skipped=ingest_result.skipped,
        next_start_index=next_start_index,
        total_results=total_results,
        exhausted=exhausted,
        success=success,
        started_at=started_at,
        finished_at=finished_at,
        errors=errors,
        metadata=metadata,
    )
