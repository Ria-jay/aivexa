# AIVEXA Supported Targets and Providers

AIVEXA only claims support for provider interfaces that are actually
implemented and can be exercised. Unknown target types are rejected
explicitly by `aivexa.targets.factory.build_model_target`.

## Model targets (`ModelTarget`)

| Provider type (`target_type`) | Adapter | Endpoint | Auth |
|---|---|---|---|
| `ollama` | `OllamaTarget` | Ollama base URL (`/api/generate`) | none |
| `openai` / `openai-compatible` | `OpenAICompatibleTarget` | Base URL (`/chat/completions`) | optional Bearer via `api_key_env` |

The OpenAI-compatible adapter works with any endpoint that implements
`POST {base}/chat/completions`, including OpenAI, vLLM, llama.cpp
server, and LM Studio.

Capabilities that cannot be determined from a text-generation probe
(such as RAG, tools, agents, multi-turn) are reported as `False` by
`TargetProfiler` rather than assumed.

### Model evaluation commands

```bash
# Ollama
aivexa model-safety run --model llama3.2:3b

# OpenAI-compatible endpoint with an API key from the environment
export MY_API_KEY=...
aivexa model-safety run \
  --target-type openai-compatible \
  --endpoint https://api.example.test/v1 \
  --model some-model \
  --api-key-env MY_API_KEY
```

## Application / agent targets (`AIApplicationTarget`)

| Adapter | Use |
|---|---|
| `ProfileHTTPApplicationTarget` | Generic authorized HTTP AI application driven by a JSON `TargetProfile` |
| `HTTPAIApplicationTarget` | Programmatic HTTP target with custom request/response mapping |
| `GeminiTarget` | Google Gemini REST `generateContent` (application adapter) |
| `MockAIApplicationTarget` | Deterministic local target for controlled tests |

Application targets implement `interact(...) -> SystemInteraction`,
which carries output, retrieved context, tool calls, authorization
context, metadata, and observations.

## Credentials

- Secrets are read from environment variables only
  (`TargetConfig.api_key_env`, `TargetProfile` `${VAR}` templates,
  `GEMINI_API_KEY`).
- No credentials are stored in code, configuration committed to the
  repository, or reports.

## Not supported (and not claimed)

- Providers without an implemented adapter are rejected.
- Live CVE/advisory feeds are not bundled; `HttpCorpusSource` provides
  the retrieval interface but requires a reachable, correctly formatted
  source and does not claim freshness without a successful fetch.
- No cloud deployment, dashboard, or GUI is provided.
