from __future__ import annotations

import json

from aivexa.cli.main import main


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
        main(
            ["target", "list", "--config", str(config)]
        )
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

    assert (
        main(
            ["target", "list", "--config", str(config)]
        )
        == 0
    )

    assert "No targets are configured" in capsys.readouterr().out


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
