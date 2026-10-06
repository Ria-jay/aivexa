import json

import pytest

from aivexa.targets.profile import (
    TargetProfile,
    extract_path,
    resolve_templates,
)


def test_target_profile_loads_and_resolves_environment(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AIVEXA_TEST_TOKEN",
        "secret-value",
    )

    path = tmp_path / "target.json"

    path.write_text(
        json.dumps(
            {
                "name": "test-app",
                "endpoint": "http://127.0.0.1:9999/chat",
                "authorized": True,
                "authorization_note": "Owned test app",
                "headers": {
                    "Authorization": (
                        "Bearer ${AIVEXA_TEST_TOKEN}"
                    )
                },
                "request_body": {
                    "message": "${input}"
                },
            }
        )
    )

    profile = TargetProfile.load(str(path))

    assert profile.name == "test-app"
    assert profile.authorized is True
    assert profile.request_headers("hello")[
        "Authorization"
    ] == "Bearer secret-value"

    assert profile.request_payload("hello") == {
        "message": "hello"
    }


def test_unauthorized_profile_is_rejected(tmp_path):
    path = tmp_path / "target.json"

    path.write_text(
        json.dumps(
            {
                "name": "unauthorized",
                "endpoint": "http://127.0.0.1:9999/chat",
                "authorized": False,
                "authorization_note": "Not authorized",
                "request_body": {
                    "message": "${input}"
                },
            }
        )
    )

    profile = TargetProfile.load(str(path))

    assert profile.authorized is False


def test_extract_path():
    payload = {
        "output": {
            "text": "hello",
        },
        "debug": {
            "items": [
                {"id": 1},
                {"id": 2},
            ]
        },
    }

    assert extract_path(
        payload,
        "output.text",
    ) == "hello"

    assert extract_path(
        payload,
        "debug.items.1.id",
    ) == 2

    assert extract_path(
        payload,
        "missing.path",
    ) is None


def test_missing_environment_variable_is_rejected():
    with pytest.raises(RuntimeError):
        resolve_templates(
            {
                "Authorization": "Bearer ${MISSING_AIVEXA_TOKEN}"
            },
            "hello",
        )
