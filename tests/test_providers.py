from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aivexa.targets.config import TargetConfig
from aivexa.targets.factory import (
    UnsupportedTargetTypeError,
    build_model_target,
    check_prerequisites,
)
from aivexa.targets.openai_compatible import (
    OpenAICompatibleTarget,
)
from aivexa.targets.profiling import TargetProfiler


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Hello from the model.",
                    }
                }
            ]
        }

        body = json.dumps(payload).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type", "application/json"
        )
        self.send_header(
            "Content-Length", str(len(body))
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=httpd.serve_forever,
        daemon=True,
    )
    thread.start()

    yield f"http://127.0.0.1:{httpd.server_address[1]}"

    httpd.shutdown()


def test_factory_builds_ollama_target():
    config = TargetConfig(
        name="local",
        target_type="ollama",
        endpoint="http://127.0.0.1:11434",
        model="llama3.2:3b",
    )

    target = build_model_target(config)

    assert target.model == "llama3.2:3b"


def test_factory_rejects_unknown_type():
    config = TargetConfig(
        name="unknown",
        target_type="not-a-provider",
        endpoint="http://127.0.0.1:9999",
        model="x",
    )

    with pytest.raises(UnsupportedTargetTypeError):
        build_model_target(config)


def test_check_prerequisites_reports_missing_key(
    monkeypatch,
):
    monkeypatch.delenv("AIVEXA_TEST_KEY", raising=False)

    config = TargetConfig(
        name="remote",
        target_type="openai",
        endpoint="https://api.example.test/v1",
        model="gpt-x",
        api_key_env="AIVEXA_TEST_KEY",
    )

    problems = check_prerequisites(config)

    assert any(
        "AIVEXA_TEST_KEY" in problem
        for problem in problems
    )


def test_openai_compatible_target_generates(server):
    target = OpenAICompatibleTarget(
        name="local-openai",
        model="test-model",
        endpoint=server,
    )

    assert target.generate("hi") == "Hello from the model."


def test_profiler_reports_reachable_and_capabilities(server):
    target = OpenAICompatibleTarget(
        name="local-openai",
        model="test-model",
        endpoint=server,
    )

    result = TargetProfiler().profile(target)

    assert result.reachable is True
    assert result.capabilities["MODEL"] is True
    assert result.capabilities["CHAT"] is True
    # Capabilities that cannot be probed are reported as False.
    assert result.capabilities["RAG"] is False
    assert result.errors == []


def test_profiler_reports_unreachable_target():
    target = OpenAICompatibleTarget(
        name="down",
        model="test-model",
        endpoint="http://127.0.0.1:1",
        timeout=1,
    )

    result = TargetProfiler(timeout=1).profile(target)

    assert result.reachable is False
    assert result.errors


class _FlakyHandler(BaseHTTPRequestHandler):
    attempts = 0

    def do_POST(self):
        type(self).attempts += 1
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        if type(self).attempts == 1:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        body = json.dumps(
            {"response": "recovered"}
        ).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type", "application/json"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


def test_openai_target_retries_transient_http_error():
    from aivexa.execution import ExecutionPolicy
    from aivexa.targets.ollama import OllamaTarget

    _FlakyHandler.attempts = 0
    httpd = HTTPServer(("127.0.0.1", 0), _FlakyHandler)
    thread = threading.Thread(
        target=httpd.serve_forever,
        daemon=True,
    )
    thread.start()

    try:
        target = OllamaTarget(
            endpoint=f"http://127.0.0.1:{httpd.server_address[1]}",
            model="test-model",
            policy=ExecutionPolicy(
                timeout=5,
                max_attempts=3,
                backoff_seconds=0.0,
                max_backoff_seconds=0.0,
            ),
        )

        # The Ollama adapter posts to /api/generate; the flaky handler
        # responds 503 once, then 200.
        output = target.generate("hi")

        assert output == "recovered"
        assert target.last_telemetry.attempts == 2
    finally:
        httpd.shutdown()
