import pytest


@pytest.fixture
def api_key():
    return "rdk-test-key-12345"


@pytest.fixture
def base_url():
    return "https://api.rodiumai.io/v1"


@pytest.fixture
def mock_response_chat():
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
def mock_response_embedding():
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
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5,
        },
    }


@pytest.fixture
def mock_response_image():
    return {
        "created": 1715000000,
        "data": [
            {
                "url": "https://api.rodiumai.io/v1/images/generated.png",
            }
        ],
    }
