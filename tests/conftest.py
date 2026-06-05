import collections

import httpx
import pytest
import respx

from rodiumai import RodiumAI


class _HttpxMockCompat:
    def __init__(self):
        self._response_queues: dict = collections.defaultdict(collections.deque)
        self._requests: list = []

    def add_response(
        self,
        url=None,
        method=None,
        json=None,
        status_code=200,
        content=None,
        headers=None,
    ):
        key = (url, method)
        kwargs: dict = {"status_code": status_code, "headers": headers or {}}
        if content is not None:
            kwargs["content"] = content
        elif json is not None:
            kwargs["json"] = json
        else:
            kwargs["content"] = b""
        resp = httpx.Response(**kwargs)
        self._response_queues[key].append(("response", resp))

    def add_exception(self, exception, url=None, method=None):
        key = (url, method)
        self._response_queues[key].append(("exception", exception))

    def _handle(self, request):
        self._requests.append(request)
        for key, queue in self._response_queues.items():
            url, method = key
            if url and url not in str(request.url):
                continue
            if method and request.method != method:
                continue
            if queue:
                kind, value = queue.popleft()
                if kind == "exception":
                    raise value
                return value
        return httpx.Response(200, json={})

    def get_request(self):
        return self._requests[-1] if self._requests else None

    def get_requests(self):
        return list(self._requests)


@pytest.fixture
def httpx_mock():
    mock = _HttpxMockCompat()
    with respx.mock:
        respx.route().mock(side_effect=mock._handle)
        yield mock
        respx.reset()


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
