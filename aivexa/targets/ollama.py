import json
from urllib import request
from urllib.error import HTTPError, URLError


class OllamaTarget:
    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    def generate(self, prompt: str, options: dict | None = None) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            "options": options or {},
            }
        ).encode("utf-8")

        req = request.Request(
            f"{self.endpoint}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.endpoint}"
            ) from exc

        return data["response"]
