# RodiumAI Python SDK

[![PyPI version](https://img.shields.io/pypi/v/rodiumai)](https://pypi.org/project/rodiumai/)
[![Python versions](https://img.shields.io/pypi/pyversions/rodiumai)](https://pypi.org/project/rodiumai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Unified AI API for Africa — access leading AI models through one SDK with local payment methods.

> **OpenAI drop-in replacement** — migrate by changing only `api_key` and `base_url`.

## Installation

```bash
pip install rodiumai
```

## Quick Start

```python
from rodiumai import RodiumAI
import os

client = RodiumAI(api_key=os.getenv("RODIUMAI_API_KEY"))

# Chat completion
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Migration from OpenAI

```python
# Before (OpenAI)
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After (RodiumAI)
from rodiumai import RodiumAI
client = RodiumAI(api_key="rdk-...")
```

That's it. Same methods, same parameters, same response types.

## Features

- **Chat Completions** — Standard & streaming (SSE)
- **Embeddings** — Single & batch text input
- **Image Generation** — Text-to-image with size/quality options
- **Audio** — Speech-to-text & text-to-speech
- **Video** — Future-ready stub
- **Retry Logic** — Automatic with exponential backoff
- **Error Hierarchy** — 8 typed errors with fix suggestions
- **Observability** — Structured JSON logging, usage stats, alerts
- **Security** — API key masked everywhere, HTTPS enforced

## Documentation

Full documentation at [docs.rodiumai.io](https://docs.rodiumai.io).

## Development

```bash
git clone https://github.com/RodiumAI/rodiumai-python.git
cd rodiumai-python
pip install -e ".[dev]"
pytest
```

## License

MIT
