from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol
from urllib import request
from urllib.error import HTTPError, URLError

from aivexa.knowledge.corpus import CorpusFormatError
from aivexa.knowledge.models import CorpusEntry


class CorpusSource(Protocol):
    """
    A source of corpus entries.

    Implementations return entries that were actually retrieved. They
    must raise a clear error when retrieval or parsing fails rather
    than returning an empty list that looks like "no data".
    """

    name: str

    def fetch(self) -> list[CorpusEntry]:
        ...


class JsonlCorpusSource:
    """Loads corpus entries from a local JSONL file."""

    def __init__(self, name: str, path: str | Path):
        self.name = name
        self.path = Path(path)

    def fetch(self) -> list[CorpusEntry]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Corpus source file not found: {self.path}"
            )

        entries: list[CorpusEntry] = []

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
                entries.append(
                    CorpusEntry.from_dict(data)
                )
            except (KeyError, TypeError) as exc:
                raise CorpusFormatError(
                    f"{self.path}:{line_number}: invalid entry: {exc}"
                ) from exc

        return entries


class HttpCorpusSource:
    """
    Fetches corpus entries from an HTTP endpoint returning JSON.

    This is a real retrieval interface, but it requires network access
    and a reachable, correctly-formatted source. It raises a clear
    error on any failure and never claims freshness without a
    successful fetch.
    """

    def __init__(
        self,
        name: str,
        url: str,
        timeout: int = 30,
        entries_key: str | None = None,
    ):
        self.name = name
        self.url = url
        self.timeout = timeout
        self.entries_key = entries_key

    def fetch(self) -> list[CorpusEntry]:
        req = request.Request(
            self.url,
            headers={"Accept": "application/json"},
            method="GET",
        )

        try:
            with request.urlopen(
                req,
                timeout=self.timeout,
            ) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(
                f"Corpus source '{self.name}' returned "
                f"HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach corpus source '{self.name}' at "
                f"{self.url}"
            ) from exc

        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CorpusFormatError(
                f"Corpus source '{self.name}' returned invalid JSON"
            ) from exc

        if self.entries_key and isinstance(data, dict):
            data = data.get(self.entries_key, [])

        if not isinstance(data, list):
            raise CorpusFormatError(
                f"Corpus source '{self.name}' did not return a list "
                "of entries"
            )

        entries: list[CorpusEntry] = []

        for item in data:
            if not isinstance(item, dict):
                raise CorpusFormatError(
                    f"Corpus source '{self.name}' returned a "
                    "non-object entry"
                )
            entries.append(CorpusEntry.from_dict(item))

        return entries
