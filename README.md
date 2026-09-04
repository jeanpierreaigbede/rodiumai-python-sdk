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
- [Streaming & real-time chat (SSE)](#streaming--real-time-chat-sse)
- [Models catalogue](#models-catalogue)
- [Embeddings](#embeddings)
- [Images (`POST /v1/images/generations`)](#images-post-v1imagesgenerations)
- [Videos (`POST /v1/videos/generations`)](#videos-post-v1videosgenerations)
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

`POST /v1/chat/completions` — OpenAI-compatible chat. All extra OpenAI fields are **pass-through** (tools, `response_format`, `seed`, etc.).

### Parameters

| Parameter | Required | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `model` | yes | string | `openai/gpt-4o` | Catalogue slug or smart alias (`rodiumai/smart`, `rodium/fast`, …) |
| `messages` | yes | array | — | `{role, content}` — string or multimodal content blocks |
| `max_tokens` | no | integer | model max | Max output tokens |
| `temperature` | no | float | — | 0–2 |
| `top_p` | no | float | — | 0–1 |
| `top_k` | no | integer | — | Gemini models |
| `stream` | no | boolean | `false` | Enable SSE streaming |
| `stop` | no | string \| array | — | Stop sequences |
| `tools` | no | array | — | OpenAI function tools |
| `tool_choice` | no | string \| object | — | `auto`, `none`, `required`, or forced function |
| `response_format` | no | object | — | `{type:"json_object"}` or `{type:"json_schema", json_schema:{...}}` |
| `session_id` | no | string | — | Custom models — conversation memory |

### Basic usage

```python
# Shorthand string
response = await client.chat("What is Rodium AI?")

# Full message list
response = await client.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain Laravel Service Providers."},
], max_tokens=500, temperature=0.5)
```

### Multi-turn conversation (history)

```python
history = [
    {"role": "user", "content": "My name is Amina."},
    {"role": "assistant", "content": "Nice to meet you, Amina!"},
    {"role": "user", "content": "What is my name?"},
]
response = await client.chat(history)
```

### Multimodal — vision (image)

```python
import base64

with open("invoice.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = await client.chat([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract the total amount from this invoice."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    },
], model="openai/gpt-4o")
```

HTTP(S) image URLs work in **chat** (unlike image/video generation endpoints).

### Multimodal — video & audio in chat

```python
# Video frame analysis
response = await client.chat([{
    "role": "user",
    "content": [
        {"type": "text", "text": "Summarize this clip."},
        {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
    ],
}])

# Inline audio (base64)
response = await client.chat([{
    "role": "user",
    "content": [
        {"type": "text", "text": "Transcribe and summarize."},
        {"type": "input_audio", "input_audio": {"data": "<BASE64>", "format": "mp3"}},
    ],
}])
```

### Function calling (tools)

```python
response = await client.chat(
    [{"role": "user", "content": "What's the weather in Lomé?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    tool_choice="auto",
)

# Handle tool_calls, call your function, send tool result back:
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    follow_up = await client.chat([
        {"role": "user", "content": "What's the weather in Lomé?"},
        response.choices[0].message,
        {"role": "tool", "tool_call_id": tool_call.id, "content": '{"temp_c": 32}'},
    ])
```

### Structured JSON output

```python
response = await client.chat(
    "List 3 African capitals as JSON.",
    response_format={"type": "json_object"},
)
```

### Smart routing

| Model alias | Behavior |
|-------------|----------|
| `rodiumai/smart`, `smart` | LLM router → 2 hops, metadata in `routing` |
| `rodium/auto`, `rodium/basic`, `rodium/fast`, `rodium/pro`, `rodium/max` | Rule-based profiles (no router hop) |

```python
response = await client.model("rodiumai/smart").chat("Summarize RODI credits.")
print(response.routing)   # requested + resolved model
print(response.cost_rodi) # RODI debited
```

See [Smart routing guide](https://www.rodiumai.io/docs/guides/smart).

### Response shape

```python
response.id              # chatcmpl-...
response.choices[0].message.content
response.choices[0].finish_reason  # stop, tool_calls, length, …
response.usage           # prompt_tokens, completion_tokens, total_tokens
response.cost_rodi       # RodiumAI extension
response.routing         # Smart routing metadata (if applicable)
response.raw             # Full upstream JSON
```

### OpenAI nested (drop-in)

```python
response = await client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## Streaming & real-time chat (SSE)

**No WebSocket** — real-time discussion uses **Server-Sent Events** (`stream: true`).

### Basic streaming

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

### Build a real-time chat loop

Pattern for a conversational UI — accumulate history client-side, stream each reply:

```python
history: list[dict] = []

async def ask(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    parts: list[str] = []
    async for delta in client.stream(history, model="openai/gpt-4o"):
        parts.append(delta)
        print(delta, end="", flush=True)  # push to UI in real time
    assistant = "".join(parts)
    history.append({"role": "assistant", "content": assistant})
    return assistant

await ask("Bonjour!")
await ask("Rappelle-moi ma première question.")  # uses history
```

### SSE wire format

```
data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}

data: [DONE]
```

Smart routing adds response headers: `X-RodiumAI-Routing`, `X-RodiumAI-Selected-Model`.

### Streaming with tools

Pass `tools` and `tool_choice` like non-streaming. Tool call deltas arrive in `delta.tool_calls`. For a simple UX, use non-streaming when tools are enabled.

### Timeouts

Use `stream_timeout=600` on the client (default) for long generations. Pass `timeout=120` per request if needed.

---

## Models catalogue

| Route | Auth | SDK |
|-------|------|-----|
| `GET /v1/models` | public | `await client.models()` |
| `GET /v1/models/{id}` | public | `await client.model_info("openai/gpt-4o")` |
| `GET /v1/models/coding` | public | `await client.coding_models()` |

```python
catalogue = await client.models()
info = await client.model_info("openai/gpt-4o")
coding = await client.coding_models()
```

Each model includes RodiumAI extensions:

| Field | Description |
|-------|-------------|
| `rodiumai_provider` | Upstream provider |
| `rodiumai_display_name` | Human label |
| `rodiumai_pricing` | RODI rates (`input_per_1m`, `output_per_1m`, `per_image`, …) |
| `rodiumai_capabilities` | `context_window`, `max_output_tokens`, modalities, `supports_streaming`, `supports_tools`, `supports_vision`, … |
| `rodiumai_kind` | `smart_router`, `smart_profile`, or standard |

Use catalogue IDs (`openai/gpt-4o`, `anthropic/claude-sonnet-4-6`) — not legacy `auto`.

---

## Embeddings

`POST /v1/embeddings`

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `model` | yes | string | e.g. `openai/text-embedding-3-small` |
| `input` | yes | string \| array | One or many texts |
| `dimensions` | no | integer | Output dimensionality (Gemini) |
| `encoding_format` | no | string | Pass-through if upstream supports |

```python
# Single text
result = await client.embeddings("Hello world", model="openai/text-embedding-3-small")
vector = result.data[0].embedding

# Batch
result = await client.embeddings(
    ["First sentence", "Second sentence"],
    model="openai/text-embedding-3-small",
)
```

Response: `{object, data: [{embedding, index}], model, usage}`.

---

## Images (`POST /v1/images/generations`)

Text-to-image, image-to-image, and inpainting depending on model.

### Parameters

| Parameter | Required | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `model` | yes | string | — | e.g. `openai/gpt-image-1`, `google/gemini-*-flash-image`, `google/imagen-*` |
| `prompt` | yes | string | — | Generation or edit instruction |
| `n` | no | integer | `1` | Count (1–10; Imagen max 4) |
| `size` | no | string | `1024x1024` | `1024x1024`, `1536x1024`, `1024x1536`, `1792x1024`, `1024x1792` |
| `quality` | no | string | `medium` | Affects RODI pricing quote |
| `aspect_ratio` | no | string | — | Gemini native (`16:9`, `1:1`, …) |
| `image` | no | object \| string | — | Reference image for edit/i2i |
| `images` | no | array | — | Up to **14** reference images (Gemini) |
| `mask` | no | object \| string | — | Inpainting mask (OpenAI edit path) |
| `background` | no | string | — | OpenAI GPT Image edit |
| `output_format` | no | string | — | OpenAI edit |
| `input_fidelity` | no | string | — | OpenAI edit |

**Image input formats** (generation endpoints — **no remote HTTP fetch**):
- Data URL: `data:image/png;base64,...`
- Raw base64 string or `{ "b64_json": "...", "mime_type": "image/png" }`
- `{ "url": "gs://bucket/object" }` (Gemini/GCS only)

**Model restrictions:**
- **Imagen** — text prompt only (no `image`/`images`)
- **OpenAI/Azure gpt-image** — edits require base64/data URL (no `gs://`)
- **Gemini Image** — multi-reference i2i (max 14 images)

### Text-to-image

```python
image = await client.images(
    model="openai/gpt-image-1",
    prompt="A red fox in the snow, illustration style",
    n=1,
    size="1024x1024",
    quality="medium",
)
b64 = image.data[0].b64_json
```

### Image-to-image (Gemini)

```python
import base64

with open("product.png", "rb") as f:
    ref = base64.b64encode(f.read()).decode()

image = await client.images(
    model="google/gemini-3.1-flash-image",
    prompt="Place on a soft white studio background",
    image={"b64_json": ref, "mime_type": "image/png"},
    images=[{"b64_json": ref}],  # additional refs (up to 14)
)
```

### Inpainting with mask (OpenAI)

```python
image = await client.images(
    model="openai/gpt-image-1",
    prompt="Replace the sky with a sunset",
    image={"b64_json": "<SOURCE_B64>"},
    mask={"b64_json": "<MASK_B64>"},
    size="1024x1024",
)
```

### Response

```python
image.created
image.data[0].b64_json      # always normalized (GPT Image always b64)
image.data[0].mime_type     # e.g. image/png
image.data[0].revised_prompt
# usage with token breakdown when available
```

---

## Videos (`POST /v1/videos/generations`)

Text-to-video and image-to-video. Jobs can take **several minutes** — always set `timeout=600`.

### Parameters

| Parameter | Required | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `model` | yes | string | — | e.g. `google/veo-3.1-generate-preview`, OpenAI Sora slugs |
| `prompt` | yes | string | — | Scene description |
| `duration_seconds` | no | number | `8` | Aliases: `seconds`, `durationSeconds` |
| `aspect_ratio` | no | string | — | e.g. `16:9`, `9:16` (mapped to Sora `size`) |
| `size` | no | string | Sora: `1280x720` | Exact `WxH` for Sora |
| `resolution` | no | string | — | Veo only |
| `resize_mode` | no | string | — | Veo only |
| `image` | no | object \| string | — | Start frame (image→video). Aliases: `input_image`, `image_url` |
| `last_frame` | no | object \| string | — | End frame interpolation (Veo). Aliases: `lastFrame`, `last_image` |

**Image formats** (same rules as images — **no `https://` fetch**):
- Data URL, raw base64, `{b64_json, mime_type}`, or `{url: "gs://..."}` for Veo

**Provider notes:**
- **Veo** — async long-running job, polled by gateway (~600 s max)
- **Sora** — duration snapped to **4, 8, or 12** seconds; start image cropped to exact `size`

### Text-to-video (simple)

```python
video = await client.videos(
    model="google/veo-3.1-generate-preview",
    prompt="Ocean waves at golden hour, cinematic",
    duration_seconds=8,
    timeout=600,
)
print(video.data[0].url or video.data[0].b64_json)
print(video.data[0].mime_type)  # video/mp4
```

### Image-to-video

```python
import base64

with open("storyboard.png", "rb") as f:
    frame = base64.b64encode(f.read()).decode()

video = await client.videos(
    model="google/veo-3.1-generate-preview",
    prompt="Subtle heartbeat pulse, soft glow",
    duration_seconds=8,
    image={"b64_json": frame, "mime_type": "image/png"},
    timeout=600,
)
```

### Interpolation (start + end frame, Veo)

```python
video = await client.videos(
    model="google/veo-3.1-generate-preview",
    prompt="Smooth morph between frames",
    duration_seconds=8,
    image={"b64_json": "<START_B64>"},
    last_frame={"b64_json": "<END_B64>"},
    timeout=600,
)
```

### Sora with aspect ratio

```python
video = await client.videos(
    model="openai/sora-...",
    prompt="Drone shot over Lomé coastline",
    duration_seconds=8,
    aspect_ratio="16:9",
    image={"b64_json": "<OPTIONAL_START_FRAME>"},
    timeout=600,
)
```

### Response

```python
{
  "created": 1746458400,
  "data": [{
    "url": "gs://... or https://...",
    "b64_json": "<mp4 base64>",
    "mime_type": "video/mp4",
    "duration_seconds": 8
  }],
  "model": "google/veo-3.1-generate-preview"
}
```

Billing: RODI charged per **actual second** generated.

---

## Audio

### Transcriptions — `POST /v1/audio/transcriptions`

Multipart upload. Auth: `Authorization: Bearer` **or** `x-api-key`.

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `file` | yes | file | Audio file (mp3, wav, m4a, …) |
| `model` | yes | string | e.g. `google/gemini-2.5-flash`, `openai/whisper-1` |
| `language` | no | string | ISO-639-1 hint (`fr`, `en`, …) |
| `prompt` | no | string | Style/spelling guide |
| `response_format` | no | string | `json`, `text`, `verbose_json` |
| `temperature` | no | float | 0–1 if supported |

```python
# From file path
transcript = await client.transcribe(
    "recording.mp3",
    model="google/gemini-2.5-flash",
    language="fr",
    prompt="Technical terms: RodiumAI, RODI, Mobile Money",
)
print(transcript.text)

# From bytes / file-like
with open("meeting.wav", "rb") as f:
    transcript = await client.transcribe(f, model="openai/whisper-1", response_format="verbose_json")
```

Response includes `usage` with `input_audio_tokens` when available.

### Speech (TTS) — `POST /v1/audio/speech`

JSON body — response is **raw audio bytes** (not JSON).

| Parameter | Required | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `model` | yes | string | — | e.g. `openai/tts-1`, `google/gemini-2.5-flash-preview-tts` |
| `input` | yes | string | — | Text to synthesize |
| `voice` | no | string | `alloy` | OpenAI: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `response_format` | no | string | — | `mp3`, `opus`, `wav`, `pcm` |
| `speed` | no | float | — | 0.25–4.0 (OpenAI) |
| `instructions` | no | string | — | Delivery style if model supports |

```python
audio_bytes = await client.speech(
    model="openai/tts-1",
    input="Hello from RodiumAI",
    voice="nova",
    response_format="mp3",
    speed=1.0,
)
open("speech.mp3", "wb").write(audio_bytes)
```

Nested:

```python
transcript = await client.audio.transcriptions.create(
    model="google/gemini-2.5-flash",
    file=open("audio.mp3", "rb"),
    language="fr",
)
speech = await client.audio.speech.create(
    model="openai/tts-1", input="Hello", voice="alloy", response_format="mp3",
)
```

---

## Anthropic Messages

`POST /v1/messages` — native Anthropic format (also works with Anthropic SDK pointing at RodiumAI).

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `model` | yes | string | e.g. `anthropic/claude-sonnet-4-6` |
| `messages` | yes | array | Anthropic message blocks |
| `max_tokens` | yes | integer | Required by Anthropic API |
| `system` | no | string \| array | System prompt |
| `stream` | no | boolean | Anthropic SSE events |
| `temperature` | no | number | |
| `top_p` | no | number | |
| `stop_sequences` | no | array | |
| `tools` | no | array | Anthropic tool definitions |
| `metadata` | no | object | |

```python
result = await client.messages(
    model="anthropic/claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a concise assistant.",
    messages=[{"role": "user", "content": "Explain RODI credits."}],
)
print(result["content"][0]["text"])
```

### Streaming (Anthropic SSE)

```python
stream = await client.messages(
    model="anthropic/claude-sonnet-4-6",
    max_tokens=1024,
    stream=True,
    messages=[{"role": "user", "content": "Write a haiku about AI."}],
)
async for event in stream:
    # event types: message_start, content_block_delta, message_delta, message_stop
    ...
```

SDK sends `anthropic-version: 2023-06-01` automatically.

---

## Wallet & pricing

RodiumAI-specific extensions (not in OpenAI SDK).

### `GET /v1/wallet`

```python
wallet = await client.wallet()
# wallet["balance_rodi"], wallet["reserved_rodi"], wallet["total_spent_rodi"], …
```

### `GET /v1/pricing`

```python
all_pricing = await client.pricing()
one_model = await client.pricing(model="openai/gpt-4o")
# Returns rodiumai_pricing + capabilities per model
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
