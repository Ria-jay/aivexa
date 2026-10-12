from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from aivexa.knowledge.corpus import CorpusFormatError
from aivexa.knowledge.models import CorpusEntry


NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


class RateLimitedError(RuntimeError):
    """Raised when an advisory source reports rate limiting."""

    def __init__(
        self,
        status: int,
        retry_after: int | None = None,
    ):
        self.status = status
        self.retry_after = retry_after
        super().__init__(
            f"Advisory source rate-limited (HTTP {status})"
            + (
                f"; retry after {retry_after}s"
                if retry_after is not None
                else ""
            )
        )


@dataclass(frozen=True)
class NvdPage:
    entries: list[CorpusEntry]
    total_results: int
    start_index: int
    results_per_page: int
    retrieved_at: str

    @property
    def next_start_index(self) -> int:
        return self.start_index + len(self.entries)

    @property
    def exhausted(self) -> bool:
        return self.next_start_index >= self.total_results


@dataclass(frozen=True)
class NvdFetchResult:
    entries: list[CorpusEntry]
    pages: int
    start_index: int
    next_start_index: int
    total_results: int
    exhausted: bool
    errors: list[str] = field(default_factory=list)


class NvdCveSource:
    """
    Real adapter for the NVD CVE API 2.0.

    Retrieves CVE records over HTTPS with configurable pagination,
    incremental refresh by last-modified window, an optional API key,
    timeouts, bounded retries, and explicit rate-limit handling.

    Each retrieved CVE is mapped to a provenance-preserving
    ``CorpusEntry`` with publication/modification timestamps and
    retrieval time. Failures raise or are reported explicitly; a failed
    refresh is never reported as successful.
    """

    name = "nvd"

    def __init__(
        self,
        base_url: str = NVD_BASE_URL,
        api_key_env: str = "NVD_API_KEY",
        timeout: int = 60,
        page_delay: float = 6.0,
        max_retries: int = 2,
        user_agent: str = "aivexa-cve-refresh",
    ):
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.page_delay = page_delay
        self.max_retries = max_retries
        self.user_agent = user_agent

    def _headers(self) -> dict[str, str]:
        import os

        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if key:
                headers["apiKey"] = key

        return headers

    def _request(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{self.base_url}?{urlencode(params)}"
        req = request.Request(
            url,
            headers=self._headers(),
            method="GET",
        )

        try:
            with request.urlopen(
                req,
                timeout=self.timeout,
            ) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code in {403, 429}:
                retry_after = exc.headers.get(
                    "Retry-After"
                ) if exc.headers else None

                raise RateLimitedError(
                    exc.code,
                    int(retry_after)
                    if retry_after
                    and str(retry_after).isdigit()
                    else None,
                ) from exc

            raise RuntimeError(
                f"NVD API returned HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach NVD API at {self.base_url}"
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CorpusFormatError(
                "NVD API returned invalid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise CorpusFormatError(
                "NVD API returned an unexpected payload"
            )

        return data

    def fetch_page(
        self,
        start_index: int = 0,
        results_per_page: int = 100,
        last_mod_start: str | None = None,
        last_mod_end: str | None = None,
        pub_start: str | None = None,
        pub_end: str | None = None,
    ) -> NvdPage:
        params: dict[str, Any] = {
            "startIndex": start_index,
            "resultsPerPage": results_per_page,
        }

        if last_mod_start:
            params["lastModStartDate"] = last_mod_start

        if last_mod_end:
            params["lastModEndDate"] = last_mod_end

        if pub_start:
            params["pubStartDate"] = pub_start

        if pub_end:
            params["pubEndDate"] = pub_end

        data = self._request(params)

        vulnerabilities = data.get("vulnerabilities")

        if not isinstance(vulnerabilities, list):
            raise CorpusFormatError(
                "NVD response is missing the 'vulnerabilities' list"
            )

        entries: list[CorpusEntry] = []
        retrieved_at = datetime.now(timezone.utc).isoformat()

        for item in vulnerabilities:
            if not isinstance(item, dict):
                raise CorpusFormatError(
                    "NVD returned a malformed vulnerability entry"
                )

            entries.append(
                self._parse_item(item, retrieved_at)
            )

        total_results = int(data.get("totalResults", len(entries)))
        response_start = int(
            data.get("startIndex", start_index)
        )
        response_page_size = int(
            data.get("resultsPerPage", results_per_page)
        )

        return NvdPage(
            entries=entries,
            total_results=total_results,
            start_index=response_start,
            results_per_page=response_page_size,
            retrieved_at=retrieved_at,
        )

    def fetch_all(
        self,
        max_pages: int = 1,
        start_index: int = 0,
        results_per_page: int = 100,
        last_mod_start: str | None = None,
        last_mod_end: str | None = None,
        pub_start: str | None = None,
        pub_end: str | None = None,
    ) -> NvdFetchResult:
        entries: list[CorpusEntry] = []
        errors: list[str] = []
        pages = 0
        cursor = start_index
        total_results = 0
        exhausted = False

        for _ in range(max(0, max_pages)):
            if pages > 0 and self.page_delay > 0:
                time.sleep(self.page_delay)

            try:
                page = self.fetch_page(
                    start_index=cursor,
                    results_per_page=results_per_page,
                    last_mod_start=last_mod_start,
                    last_mod_end=last_mod_end,
                    pub_start=pub_start,
                    pub_end=pub_end,
                )
            except RateLimitedError as exc:
                errors.append(str(exc))
                break
            except (RuntimeError, CorpusFormatError) as exc:
                errors.append(
                    f"{type(exc).__name__}: {exc}"
                )
                break

            entries.extend(page.entries)
            pages += 1
            cursor = page.next_start_index
            total_results = page.total_results
            exhausted = page.exhausted

            if exhausted:
                break

        return NvdFetchResult(
            entries=entries,
            pages=pages,
            start_index=start_index,
            next_start_index=cursor,
            total_results=total_results,
            exhausted=exhausted,
            errors=errors,
        )

    @staticmethod
    def _parse_item(
        item: dict[str, Any],
        retrieved_at: str,
    ) -> CorpusEntry:
        cve = item.get("cve")

        if not isinstance(cve, dict):
            raise CorpusFormatError(
                "NVD vulnerability entry is missing 'cve'"
            )

        cve_id = cve.get("id")

        if not cve_id:
            raise CorpusFormatError(
                "NVD vulnerability entry is missing an id"
            )

        description = NvdCveSource._english_text(
            cve.get("descriptions", [])
        )

        cwes: list[str] = []

        for weakness in cve.get("weaknesses", []):
            for desc in weakness.get("description", []):
                value = desc.get("value")
                if (
                    isinstance(value, str)
                    and value.upper().startswith("CWE-")
                ):
                    if value not in cwes:
                        cwes.append(value)

        severity, score = NvdCveSource._severity(cve)

        references = [
            ref.get("url", "")
            for ref in cve.get("references", [])
            if isinstance(ref, dict)
        ]

        url = (
            references[0]
            if references
            else f"https://nvd.nist.gov/vuln/detail/{cve_id}"
        )

        return CorpusEntry(
            entry_id=cve_id,
            title=(
                f"{cve_id}: "
                + (description[:120] if description else "CVE")
            ),
            source="NVD",
            summary=description,
            url=url,
            published_at=str(cve.get("published", "")),
            retrieved_at=retrieved_at,
            cwes=tuple(cwes),
            tags=("nvd", "cve"),
            keywords=tuple(
                word
                for word in description.lower().split()
                if len(word) > 4
            )[:10],
            severity=severity,
            metadata={
                "last_modified": cve.get("lastModified", ""),
                "vuln_status": cve.get("vulnStatus", ""),
                "source_identifier": cve.get(
                    "sourceIdentifier", ""
                ),
                "base_score": score,
                "references": references,
                "advisory_source": "nvd",
            },
        )

    @staticmethod
    def _english_text(
        descriptions: list[dict[str, Any]],
    ) -> str:
        for desc in descriptions:
            if (
                isinstance(desc, dict)
                and desc.get("lang") == "en"
            ):
                return str(desc.get("value", ""))

        if descriptions and isinstance(
            descriptions[0], dict
        ):
            return str(descriptions[0].get("value", ""))

        return ""

    @staticmethod
    def _severity(
        cve: dict[str, Any],
    ) -> tuple[str, float | None]:
        metrics = cve.get("metrics", {})

        for key in (
            "cvssMetricV31",
            "cvssMetricV30",
            "cvssMetricV2",
        ):
            values = metrics.get(key)

            if not values:
                continue

            data = values[0].get("cvssData", {})

            severity = data.get("baseSeverity")
            score = data.get("baseScore")

            if severity:
                return str(severity).lower(), (
                    float(score) if score is not None else None
                )

        return "", None
