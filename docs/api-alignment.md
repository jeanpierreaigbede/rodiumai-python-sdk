# API alignment with Rodium AI

This package implements the [Rodium AI REST API](https://www.rodiumai.io/docs/api/overview). The API is **OpenAI-compatible**: same paths, JSON shapes, and Bearer authentication.

**Base URL:** `RODIUMAI_BASE_URL` env or constructor `base_url` (default `https://api.rodiumai.io/v1`).

## Dual API surface

| Style | Example |
|-------|---------|
| Flat (Laravel parity) | `await client.chat("Hello")`, `await client.models()`, `await client.wallet()` |
| Fluent builder | `await client.model("openai/gpt-4o").temperature(0.7).chat("Hi")` |
| OpenAI nested | `await client.chat.completions.create(...)`, `await client.embeddings.create(...)` |

## Endpoints implemented

| Official endpoint | Flat method | Nested |
|-------------------|-------------|--------|
| `POST /v1/chat/completions` | `chat()` | `chat.completions.create()` |
| `POST /v1/chat/completions` (stream) | `stream()` | `chat.completions.create(stream=True)` |
| `GET /v1/models` | `models()` | `models.list()` |
| `GET /v1/models/{id}` | `model_info()` | `models.retrieve()` |
| `GET /v1/models/coding` | `coding_models()` | `models.list_coding()` |
| `POST /v1/embeddings` | `embeddings()` | `embeddings.create()` |
| `POST /v1/images/generations` | `images()` | `images.generate()` |
| `POST /v1/videos/generations` | `videos()` | `video.generations.create()` |
| `POST /v1/audio/transcriptions` | `transcribe()` | `audio.transcriptions.create()` |
| `POST /v1/audio/speech` | `speech()` (bytes) | `audio.speech.create()` |
| `POST /v1/messages` | `messages()` | `messages.create()` |
| `GET /v1/wallet` | `wallet()` | — |
| `GET /v1/pricing` | `pricing()` | — |

## Defaults & errors

- Default model: `openai/gpt-4o` (`RODIUMAI_DEFAULT_MODEL` override)
- API keys: `rd_sk_...` or legacy `rdk-...`
- 402 error code: `insufficient_balance` (alias `InsufficientBalanceError`)
- Anthropic + OpenAI error envelopes parsed in `_http.py`
- Speech returns binary; transcriptions use multipart upload

See the [Laravel SDK alignment doc](../rodiumai-laravel-sdk/docs/api-alignment.md) for full parameter passthrough and response extensions (`cost_rodi`, `routing`, etc.).
