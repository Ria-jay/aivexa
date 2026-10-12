from __future__ import annotations

import json
from pathlib import Path

from aivexa.knowledge.corpus import (
    CorpusFormatError,
    KnowledgeCorpus,
)
from aivexa.knowledge.models import CorpusEntry
from aivexa.knowledge.sources import JsonlCorpusSource


def _entry(entry_id="CWE-79", title="XSS", **kwargs):
    return CorpusEntry(
        entry_id=entry_id,
        title=title,
        source="test-fixture",
        summary=kwargs.pop(
            "summary",
            "Improper neutralization of input.",
        ),
        retrieved_at="2026-01-01T00:00:00+00:00",
        cwes=kwargs.pop("cwes", ("CWE-79",)),
        tags=kwargs.pop("tags", ("web",)),
        **kwargs,
    )


def test_corpus_add_dedup_and_update(tmp_path):
    corpus = KnowledgeCorpus(tmp_path / "corpus.jsonl")

    assert corpus.add(_entry()) is True
    assert corpus.add(_entry()) is False
    assert len(corpus) == 1

    changed = _entry(summary="Updated summary.")
    assert corpus.add(changed) is True
    assert corpus.get("CWE-79").summary == "Updated summary."


def test_corpus_ingest_counts(tmp_path):
    corpus = KnowledgeCorpus(tmp_path / "corpus.jsonl")

    result = corpus.ingest([_entry(), _entry("CWE-89", "SQLi")])

    assert result.added == 2
    assert result.updated == 0

    result2 = corpus.ingest([_entry(), _entry("CWE-89", "SQLi")])

    assert result2.unchanged == 2


def test_corpus_search_and_persist(tmp_path):
    path = tmp_path / "corpus.jsonl"
    corpus = KnowledgeCorpus(path)

    corpus.ingest(
        [
            _entry("CWE-79", "Cross-site scripting"),
            _entry(
                "CWE-89",
                "SQL injection",
                summary="Improper SQL command neutralization.",
                tags=("database",),
                cwes=("CWE-89",),
            ),
        ]
    )
    corpus.save()

    reloaded = KnowledgeCorpus(path)

    assert len(reloaded) == 2

    results = reloaded.search(query="sql")
    assert [entry.entry_id for entry in results] == ["CWE-89"]

    by_tag = reloaded.search(tag="web")
    assert [entry.entry_id for entry in by_tag] == ["CWE-79"]

    by_cwe = reloaded.search(cwe="cwe-89")
    assert [entry.entry_id for entry in by_cwe] == ["CWE-89"]


def test_jsonl_source_and_format_error(tmp_path):
    good = tmp_path / "good.jsonl"
    entry = _entry()
    good.write_text(
        json.dumps(entry.to_dict()) + "\n",
        encoding="utf-8",
    )

    entries = JsonlCorpusSource("fixture", good).fetch()
    assert len(entries) == 1

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")

    try:
        JsonlCorpusSource("fixture", bad).fetch()
    except CorpusFormatError:
        pass
    else:
        raise AssertionError("Expected CorpusFormatError")


def test_staleness(tmp_path):
    from datetime import datetime, timezone

    corpus = KnowledgeCorpus(tmp_path / "corpus.jsonl")
    entry = _entry()
    corpus.add(entry, retrieved_at="2020-01-01T00:00:00+00:00")

    stored = corpus.get("CWE-79")

    assert corpus.is_stale(
        stored,
        max_age_days=30,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
