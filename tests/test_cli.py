from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aivexa.cli.main import main


class _OpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)

        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "I cannot help with that request."
                            )
                        }
                    }
                ]
            }
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


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    thread = threading.Thread(
        target=httpd.serve_forever,
        daemon=True,
    )
    thread.start()

    yield f"http://127.0.0.1:{httpd.server_address[1]}"

    httpd.shutdown()


def test_cli_target_registry_round_trip(tmp_path, capsys):
    config = tmp_path / "targets.json"

    assert (
        main(
            [
                "target",
                "add",
                "demo",
                "--type",
                "ollama",
                "--endpoint",
                "http://127.0.0.1:11434",
                "--model",
                "llama3.2:3b",
                "--config",
                str(config),
            ]
        )
        == 0
    )

    capsys.readouterr()

    assert (
        main(["target", "list", "--config", str(config)])
        == 0
    )

    assert "demo: ollama" in capsys.readouterr().out

    assert (
        main(
            ["target", "show", "demo", "--config", str(config)]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "demo"
    assert payload["model"] == "llama3.2:3b"

    assert (
        main(
            ["target", "remove", "demo", "--config", str(config)]
        )
        == 0
    )


def test_cli_reports_errors_without_crashing(tmp_path, capsys):
    exit_code = main(
        [
            "target",
            "show",
            "missing",
            "--config",
            str(tmp_path / "absent.json"),
        ]
    )

    assert exit_code == 1
    assert "AIVEXA ERROR" in capsys.readouterr().err


def test_cli_target_profile(server, capsys):
    exit_code = main(
        [
            "target",
            "profile",
            "--model",
            "test-model",
            "--target-type",
            "openai",
            "--endpoint",
            server,
            "--json",
        ]
    )

    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["reachable"] is True
    assert payload["capabilities"]["MODEL"] is True


def test_cli_model_security_run(server, tmp_path, capsys):
    exit_code = main(
        [
            "model-security",
            "run",
            "--model",
            "test-model",
            "--target-type",
            "openai",
            "--endpoint",
            server,
            "--db",
            str(tmp_path / "sec.db"),
        ]
    )

    assert exit_code == 0
    assert (
        "AIVEXA MODEL SECURITY EVALUATION"
        in capsys.readouterr().out
    )


def test_cli_research_metrics_on_empty_db(tmp_path, capsys):
    exit_code = main(
        [
            "research",
            "metrics",
            "--db",
            str(tmp_path / "empty.db"),
        ]
    )

    assert exit_code == 0
    assert "AIVEXA RESEARCH METRICS" in capsys.readouterr().out


def test_cli_report_experiment_missing_is_error(tmp_path, capsys):
    exit_code = main(
        [
            "report",
            "experiment",
            "MISSING",
            "--db",
            str(tmp_path / "empty.db"),
        ]
    )

    assert exit_code == 1
    assert "AIVEXA ERROR" in capsys.readouterr().err
