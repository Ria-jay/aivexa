import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from aivexa.targets.http_application import (
    HTTPAIApplicationTarget,
)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(
            self.rfile.read(length).decode("utf-8")
        )

        response = {
            "system_output": (
                f"Processed: {body['user_input']}"
            ),
            "retrieved_context": [
                {
                    "source": "public-documents",
                    "scope": "public",
                }
            ],
            "tool_calls": [
                {
                    "tool": "search",
                    "arguments": {
                        "scope": "public",
                    },
                }
            ],
            "observations": [
                "HTTP target interaction completed."
            ],
            "metadata": {
                "accessed_data_sources": [
                    "public-documents"
                ],
                "accessed_data_scopes": [
                    "public"
                ],
                "execution_identity": {
                    "principal": "user-1",
                    "tenant": "tenant-a",
                },
            },
        }

        encoded = json.dumps(response).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Content-Length",
            str(len(encoded)),
        )
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def test_http_application_target_interacts():
    server = HTTPServer(
        ("127.0.0.1", 0),
        Handler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    try:
        target = HTTPAIApplicationTarget(
            name="phase4-http-test",
            endpoint=(
                f"http://127.0.0.1:"
                f"{server.server_port}/chat"
            ),
        )

        interaction = target.interact(
            experiment_id="AVX-P4-HTTP-001",
            user_input="show public information",
            context={
                "request_type": "public"
            },
            authorization_context={
                "principal": "user-1",
                "tenant": "tenant-a",
                "tools": ["search"],
                "data_sources": [
                    "public-documents"
                ],
                "data_scopes": ["public"],
            },
        )

        assert interaction.experiment_id == (
            "AVX-P4-HTTP-001"
        )
        assert interaction.system_output == (
            "Processed: show public information"
        )
        assert len(interaction.retrieved_context) == 1
        assert interaction.tool_calls[0]["tool"] == "search"
        assert interaction.metadata["http_status"] == 200
        assert interaction.metadata["implementation"] == "http"
    finally:
        server.shutdown()
        server.server_close()
