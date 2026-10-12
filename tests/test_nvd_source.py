from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from aivexa.knowledge.corpus import (
    CorpusFormatError,
    KnowledgeCorpus,
)
from aivexa.knowledge.nvd import NvdCveSource
from aivexa.knowledge.refresh import refresh_corpus


def _cve(cve_id: str, cwe: str, severity: str = "HIGH"):
    return {
        "cve": {
            "id": cve_id,
            "sourceIdentifier": "test@example.org",
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-06-01T00:00:00.000",
            "vulnStatus": "Analyzed",
            "descriptions": [
                {
                    "lang": "en",
                    "value": (
                        f"Improper handling in {cve_id} allows "
                        "untrusted input to affect behavior."
                    ),
                }
            ],
            "metrics": {
                "cvssMetricV31": [
                    {
                        "cvssData": {
                            "baseScore": 8.1,
                            "baseSeverity": severity,
                        }
                    }
                ]
            },
            "weaknesses": [
                {
                    "source": "test",
                    "type": "Primary",
                    "description": [
                        {"lang": "en", "value": cwe}
                    ],
                }
            ],
            "references": [
                {
                    "url": (
                        "https://example.org/advisory/"
                        + cve_id
                    ),
                    "source": "test",
                }
            ],
        }
    }


ALL_CVES = [
    _cve("CVE-2024-0001", "CWE-79"),
    _cve("CVE-2024-0002", "CWE-89"),
    _cve("CVE-2024-0003", "CWE-1426"),
]


class _NvdHandler(BaseHTTPRequestHandler):
    mode = "normal"
    last_query: dict = {}

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        type(self).last_query = params

        if type(self).mode == "ratelimit":
            self.send_response(429)
            self.send_header("Retry-After", "30")
            self.end_headers()
            return

        if type(self).mode == "malformed":
            body = b"{not json}"
            self.send_response(200)
            self.send_header(
                "Content-Type", "application/json"
            )
            self.send_header(
                "Content-Length", str(len(body))
            )
            self.end_headers()
            self.wfile.write(body)
            return

        start = int(params.get("startIndex", ["0"])[0])
        page_size = int(
            params.get("resultsPerPage", ["2"])[0]
        )

        page = ALL_CVES[start:start + page_size]

        payload = {
            "resultsPerPage": len(page),
            "startIndex": start,
            "totalResults": len(ALL_CVES),
            "vulnerabilities": page,
        }

        body = json.dumps(payload).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type", "application/json"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@pytest.fixture
def nvd_server():
    _NvdHandler.mode = "normal"
    httpd = HTTPServer(("127.0.0.1", 0), _NvdHandler)
    thread = threading.Thread(
        target=httpd.serve_forever,
        daemon=True,
    )
    thread.start()

    yield f"http://127.0.0.1:{httpd.server_address[1]}"

    httpd.shutdown()


def _source(server):
    return NvdCveSource(
        base_url=f"{server}/rest/json/cves/2.0",
        api_key_env="",
        timeout=10,
        page_delay=0.0,
    )


def test_fetch_page_parses_cve(nvd_server):
    page = _source(nvd_server).fetch_page(
        start_index=0,
        results_per_page=1,
    )

    assert page.total_results == 3
    assert page.start_index == 0
    assert len(page.entries) == 1

    entry = page.entries[0]
    assert entry.entry_id == "CVE-2024-0001"
    assert entry.source == "NVD"
    assert entry.cwes == ("CWE-79",)
    assert entry.severity == "high"
    assert entry.published_at.startswith("2024-01-01")
    assert entry.retrieved_at
    assert entry.metadata["last_modified"].startswith(
        "2024-06-01"
    )
    assert entry.url == (
        "https://example.org/advisory/CVE-2024-0001"
    )


def test_fetch_all_paginates(nvd_server):
    result = _source(nvd_server).fetch_all(
        max_pages=10,
        results_per_page=2,
    )

    assert result.pages == 2
    assert len(result.entries) == 3
    assert result.exhausted is True
    assert result.next_start_index == 3
    assert result.errors == []

    ids = [entry.entry_id for entry in result.entries]
    assert ids == [
        "CVE-2024-0001",
        "CVE-2024-0002",
        "CVE-2024-0003",
    ]


def test_fetch_all_respects_max_pages(nvd_server):
    result = _source(nvd_server).fetch_all(
        max_pages=1,
        results_per_page=2,
    )

    assert result.pages == 1
    assert len(result.entries) == 2
    assert result.exhausted is False
    assert result.next_start_index == 2


def test_incremental_refresh_passes_since(nvd_server):
    _source(nvd_server).fetch_all(
        max_pages=1,
        results_per_page=2,
        last_mod_start="2024-05-01T00:00:00.000",
    )

    assert (
        _NvdHandler.last_query["lastModStartDate"][0]
        == "2024-05-01T00:00:00.000"
    )


def test_rate_limit_is_reported(nvd_server):
    _NvdHandler.mode = "ratelimit"

    result = _source(nvd_server).fetch_all(max_pages=3)

    assert result.pages == 0
    assert result.entries == []
    assert any(
        "rate-limited" in error for error in result.errors
    )


def test_malformed_response_is_reported(nvd_server):
    _NvdHandler.mode = "malformed"

    source = _source(nvd_server)

    with pytest.raises(CorpusFormatError):
        source.fetch_page()

    result = source.fetch_all(max_pages=1)

    assert result.errors
    assert "CorpusFormatError" in result.errors[0]


def test_refresh_corpus_ingests_and_dedupes(
    nvd_server, tmp_path
):
    corpus = KnowledgeCorpus(tmp_path / "corpus.jsonl")
    source = _source(nvd_server)

    report = refresh_corpus(
        corpus,
        source,
        max_pages=10,
        results_per_page=2,
    )

    assert report.success is True
    assert report.added == 3
    assert len(corpus) == 3
    assert report.exhausted is True

    # Re-running the same refresh is idempotent.
    report2 = refresh_corpus(
        corpus,
        source,
        max_pages=10,
        results_per_page=2,
    )

    assert report2.added == 0
    assert report2.unchanged == 3
    assert len(corpus) == 3


def test_refresh_failure_is_not_success(nvd_server, tmp_path):
    _NvdHandler.mode = "ratelimit"

    corpus = KnowledgeCorpus(tmp_path / "corpus.jsonl")

    report = refresh_corpus(
        corpus,
        _source(nvd_server),
        max_pages=2,
    )

    assert report.success is False
    assert report.entries_fetched == 0
    assert report.errors
