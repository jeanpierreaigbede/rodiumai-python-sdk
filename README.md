# RodiumAI Python SDK

Official async Python SDK for the [Rodium AI](https://www.rodiumai.io) API — unified access to AI models (OpenAI, Anthropic, Google, DeepSeek…) with **RODI** credit billing and **Mobile Money** top-ups.

> **OpenAI-compatible** REST API: same endpoints and payloads as documented at [rodiumai.io/docs](https://www.rodiumai.io/docs).

[![PyPI version](https://img.shields.io/pypi/v/rodiumai)](https://pypi.org/project/rodiumai/)
[![Python versions](https://img.shields.io/pypi/pyversions/rodiumai)](https://pypi.org/project/rodiumai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/Docteur-Parfait/rodiumai-python-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Docteur-Parfait/rodiumai-python-sdk/actions)

## Links

| Resource | URL |
|----------|-----|
| **PyPI** | [pypi.org/project/rodiumai](https://pypi.org/project/rodiumai/) |
| **Source code** | [github.com/Docteur-Parfait/rodiumai-python-sdk](https://github.com/Docteur-Parfait/rodiumai-python-sdk) |
| **Python SDK guide** | [rodiumai.io/docs/guides/python-sdk](https://www.rodiumai.io/docs/guides/python-sdk) |
| **API documentation** | [rodiumai.io/docs](https://www.rodiumai.io/docs) |
| **Dashboard & API keys** | [rodiumai.io/dashboard](https://www.rodiumai.io/dashboard) |

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Three ways to call the API](#three-ways-to-call-the-api)
- [Quick start](#quick-start)
- [Chat completions](#chat-completions)
- [Streaming (SSE)](#streaming-sse)
- [Models](#models)
- [Embeddings](#embeddings)
- [Images & videos](#images--videos)
- [Audio](#audio)
- [Anthropic Messages](#anthropic-messages)
- [Wallet & pricing](#wallet--pricing)
- [Error handling](#error-handling)
- [SDK reference](#sdk-reference)
- [OpenAI migration](#openai-migration)
- [Local development](#local-development)
- [Migration 0.2.x → 0.3.0](#migration-02x--030)
- [Testing](#testing)
- [License](#license)

## Requirements

- Python **3.9+**
- `httpx` (installed automatically)
- Rodium AI account + API key (`rd_sk_…`): [dashboard](https://www.rodiumai.io/dashboard)

## Installation

```bash
pip install "rodiumai>=0.3"
```

## Configuration

Environment variables (recommended):

```bash
export RODIUMAI_API_KEY=rd_sk_your_secret_key
export RODIUMAI_BASE_URL=https://api.rodiumai.io/v1   # optional
export RODIUMAI_DEFAULT_MODEL=openai/gpt-4o          # optional
```

Constructor options:

```python
from rodiumai import RodiumAI

client = RodiumAI(
    api_key="rd_sk_...",           # or RODIUMAI_API_KEY env
    base_url="https://api.rodiumai.io/v1",
    default_model="openai/gpt-4o",
    timeout=30.0,                  # seconds
    stream_timeout=600.0,          # seconds (streaming / long jobs)
    max_retries=3,
)
```

| Variable / option | Default | Description |
|-------------------|---------|-------------|
| `RODIUMAI_API_KEY` | — | Secret key from the dashboard |
| `RODIUMAI_BASE_URL` | `https://api.rodiumai.io/v1` | Gateway base URL (`http://localhost:8001/v1` for local dev) |
| `RODIUMAI_DEFAULT_MODEL` | `openai/gpt-4o` | Default model slug |
| `timeout` | `30` | HTTP timeout in seconds |
| `stream_timeout` | `600` | Timeout for streams and long requests |

Never commit API keys or `.env` files.

## Three ways to call the API

| Style | When to use | Example |
|-------|-------------|---------|
| **Flat API** | Recommended — parity with Laravel SDK | `await client.chat("Hello")` |
| **Fluent builder** | Chain model / decoding options | `await client.model("openai/gpt-4o").temperature(0.7).chat("Hi")` |
| **OpenAI nested** | Drop-in for existing OpenAI code | `await client.chat.completions.create(...)` |

All three hit the same gateway endpoints.

## Quick start

```python
import asyncio
from rodiumai import RodiumAI

async def main():
    client = RodiumAI()

    # Flat API (recommended)
    response = await client.chat("Explain RODI credits in one sentence.")
    print(response.choices[0].message.content)
    print(response.cost_rodi)

    # Fluent builder
    response = await client.model("openai/gpt-4o").temperature(0.7).chat("Hello!")

asyncio.run(main())
```

## Chat completions

Aligned with [docs/api/chat-completions](https://www.rodiumai.io/docs/api/chat-completions).

```python
# Shorthand string
response = await client.chat("What is Rodium AI?")

# Full message list
response = await client.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain Laravel Service Providers."},
], max_tokens=500, temperature=0.5)

# Passthrough (tools, response_format, stop, …)
response = await client.chat(messages, tools=[...], response_format={"type": "json_object"})
```

### Smart routing

```python
response = await client.model("rodiumai/smart").chat("Summarize RODI credits.")
print(response.routing)  # resolved model metadata from gateway
```

See [Smart routing guide](https://www.rodiumai.io/docs/guides/smart).

### OpenAI nested (drop-in)

```python
response = await client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Streaming (SSE)

```python
async for delta in client.stream("Tell a short story about Lagos."):
    print(delta, end="", flush=True)
```

Nested equivalent:

```python
stream = await client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
async for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

## Models

```python
catalogue = await client.models()           # GET /v1/models
info = await client.model_info("openai/gpt-4o")  # GET /v1/models/{id}
coding = await client.coding_models()       # GET /v1/models/coding

# Nested
catalogue = await client.models.list()
info = await client.models.retrieve("openai/gpt-4o")
```

Use catalogue IDs (e.g. `openai/gpt-4o`, `anthropic/claude-sonnet-4-6`) — not legacy `auto`.

## Embeddings

```python
result = await client.embeddings("Hello world", model="openai/text-embedding-3-small")
vector = result.data[0].embedding

# Batch
result = await client.embeddings(["Hello", "World"], model="openai/text-embedding-3-small")
```

## Images & videos

```python
image = await client.images(
    model="openai/gpt-image-1",
    prompt="A sunset over Lomé",
    size="1024x1024",
)
print(image.data[0].url)

video = await client.videos(
    model="google/veo-3.1-generate-preview",
    prompt="Ocean waves at golden hour",
    duration_seconds=8,
    timeout=600,
)
print(video.data[0].url)
```

## Audio

```python
# Transcription (multipart upload — path, bytes, or file-like)
transcript = await client.transcribe("recording.mp3", model="google/gemini-2.5-flash", language="fr")
print(transcript.text)

# Text-to-speech (returns raw bytes)
audio_bytes = await client.speech(
    model="openai/tts-1",
    input="Hello from RodiumAI",
    voice="alloy",
)
open("speech.mp3", "wb").write(audio_bytes)
```

Nested:

```python
transcript = await client.audio.transcriptions.create(model="...", file=open("audio.mp3", "rb"))
speech = await client.audio.speech.create(model="...", input="Hello", voice="alloy")
```

## Anthropic Messages

Drop-in for [POST /v1/messages](https://www.rodiumai.io/docs/api/messages):

```python
result = await client.messages(
    model="anthropic/claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain RODI credits."}],
)
print(result)
```

## Wallet & pricing

RodiumAI extensions:

```python
wallet = await client.wallet()
print(wallet)

pricing = await client.pricing()
pricing_model = await client.pricing(model="openai/gpt-4o")
```

## Error handling

See [docs/api/errors](https://www.rodiumai.io/docs/api/errors).

| HTTP | Exception | `error_code` |
|------|-----------|--------------|
| 401 | `InvalidAPIKeyError` | `invalid_api_key` |
| 402 | `InsufficientRODIError` / `InsufficientBalanceError` | `insufficient_balance` |
| 403 | `PermissionDeniedError` | `permission_denied` |
| 404 | `ModelNotFoundError` | `model_not_found` |
| 429 | `RateLimitError` | `rate_limit_exceeded` — use `retry_after` |
| 500 | `InternalServerError` | `internal_error` |
| 503 | `ServiceUnavailableError` | `service_unavailable` |
| Timeout | `TimeoutError` | `timeout` |
| Network | `NetworkError` | `network_error` |

Both OpenAI-shaped and Anthropic-shaped error envelopes are parsed from the gateway.

```python
from rodiumai import RodiumAI, InsufficientBalanceError, RateLimitError

client = RodiumAI()

try:
    await client.chat("Test")
except InsufficientBalanceError as e:
    print(e.error_code, e.request_id)
except RateLimitError as e:
    await asyncio.sleep(e.retry_after or 1.0)
```

## SDK reference

| Flat method | Nested equivalent | Gateway route |
|-------------|-------------------|---------------|
| `chat(messages, **opts)` | `chat.completions.create(...)` | `POST /v1/chat/completions` |
| `stream(messages, **opts)` | `chat.completions.create(stream=True)` | `POST /v1/chat/completions` (SSE) |
| `models()` | `models.list()` | `GET /v1/models` |
| `model_info(id)` | `models.retrieve(id)` | `GET /v1/models/{id}` |
| `coding_models()` | `models.list_coding()` | `GET /v1/models/coding` |
| `embeddings(input, **opts)` | `embeddings.create(...)` | `POST /v1/embeddings` |
| `images(**opts)` | `images.generate(...)` | `POST /v1/images/generations` |
| `videos(**opts)` | `video.generations.create(...)` | `POST /v1/videos/generations` |
| `transcribe(file, **opts)` | `audio.transcriptions.create(...)` | `POST /v1/audio/transcriptions` |
| `speech(**opts)` → `bytes` | `audio.speech.create(...)` | `POST /v1/audio/speech` |
| `messages(**opts)` | `messages.create(...)` | `POST /v1/messages` |
| `wallet()` | — | `GET /v1/wallet` |
| `pricing(model?)` | — | `GET /v1/pricing` |

Fluent builder: `model()`, `temperature()`, `top_p()`, `max_tokens()`, `system_prompt()`.

Response extensions on `ChatCompletion`: `cost_rodi`, `routing`, `raw`.

Technical mapping: [docs/api-alignment.md](docs/api-alignment.md).

## OpenAI migration

```python
# Before (OpenAI)
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key="sk-...")

# After (RodiumAI) — nested API unchanged
from rodiumai import RodiumAI
client = RodiumAI(api_key="rd_sk_...")

response = await client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Or use the flat API: `await client.chat("Hello!")`.

## Local development

Point the SDK at a local gateway (e.g. Docker Compose on port 8001):

```bash
export RODIUMAI_BASE_URL=http://localhost:8001/v1
export RODIUMAI_API_KEY=rd_sk_dev_...
```

## Migration 0.2.x → 0.3.0

| Change | Action |
|--------|--------|
| Default model | `auto` → `openai/gpt-4o` (override via `RODIUMAI_DEFAULT_MODEL`) |
| Flat + fluent API | New recommended surface: `client.chat()`, `client.stream()`, … |
| Video | Implemented (`client.videos()`) — was stub in 0.2.x |
| 402 error code | `insufficient_rodi` → `insufficient_balance` |
| Speech | Returns binary bytes (not JSON envelope) |
| New resources | `models`, `messages`, `wallet`, `pricing` |

Require `rodiumai>=0.3` in your dependencies.

## Testing

```bash
git clone https://github.com/Docteur-Parfait/rodiumai-python-sdk.git
cd rodiumai-python-sdk
pip install -e ".[dev]"
pytest tests/unit
```

## License

MIT — see [LICENSE](LICENSE).
