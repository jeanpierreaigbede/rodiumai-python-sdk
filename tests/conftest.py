import pytest

from rodiumai import RodiumAI


@pytest.fixture
def api_key():
    return "rdk-test-key-12345"


@pytest.fixture
def base_url():
    return "https://api.rodiumai.io/v1"


@pytest.fixture
def client(api_key, base_url):
    return RodiumAI(api_key=api_key, base_url=base_url)


@pytest.fixture
def mock_chat_response():
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1715000000,
        "model": "auto",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_embedding_response():
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
        "model": "auto",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


@pytest.fixture
def mock_batch_embedding_response():
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        ],
        "model": "auto",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


@pytest.fixture
def mock_image_response():
    return {
        "created": 1715000000,
        "data": [
            {"url": "https://api.rodiumai.io/v1/images/generated.png"},
        ],
    }


@pytest.fixture
def mock_transcription_response():
    return {"text": "Hello, world."}


@pytest.fixture
def mock_speech_response():
    return {"content": "fake-audio-bytes", "content_type": "audio/mpeg"}
