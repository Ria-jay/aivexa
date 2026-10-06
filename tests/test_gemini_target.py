import json
from unittest.mock import patch

from aivexa.targets.gemini import GeminiTarget


class FakeResponse:
    status = 200

    def read(self):
        return json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "Hello from Gemini."
                                },
                                {
                                    "functionCall": {
                                        "name": "search",
                                        "args": {
                                            "scope": "public"
                                        },
                                    }
                                },
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "totalTokenCount": 10
                },
            }
        ).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_gemini_target_extracts_text_and_tool_calls():
    target = GeminiTarget(
        name="test-gemini",
        model="test-model",
        api_key="test-key",
    )

    with patch(
        "aivexa.targets.gemini.request.urlopen",
        return_value=FakeResponse(),
    ):
        interaction = target.interact(
            experiment_id="GEMINI-001",
            user_input="Search public information.",
            context={
                "tools": [
                    {
                        "name": "search",
                        "description": "Search public data.",
                        "parameters": {
                            "type": "object",
                        },
                    }
                ]
            },
            authorization_context={
                "tools": ["search"],
            },
        )

    assert interaction.system_output == (
        "Hello from Gemini."
    )

    assert len(interaction.tool_calls) == 1
    assert interaction.tool_calls[0]["tool"] == "search"
    assert interaction.tool_calls[0]["authorized"] is True
    assert interaction.tool_calls[0]["executed"] is False
