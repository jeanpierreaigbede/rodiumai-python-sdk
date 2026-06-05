import pytest

from rodiumai.errors import ModelNotFoundError
from rodiumai.resources.chat import ChatCompletion, ChatCompletionChunk


class TestChatIntegration:
    @pytest.mark.asyncio
    async def test_chat_completion_endpoint(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
            headers={"X-Request-ID": "req-123"},
        )
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(response, ChatCompletion)
        assert response.id == "chatcmpl-123"
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_chat_streaming_endpoint(self, monkeypatch, client):
        chunks = [
            {
                "id": "chunk-1",
                "object": "chat.completion.chunk",
                "created": 1715000000,
                "model": "auto",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            },
            {
                "id": "chunk-1",
                "object": "chat.completion.chunk",
                "created": 1715000000,
                "model": "auto",
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
            },
        ]

        async def mock_stream(*args, **kwargs):
            for c in chunks:
                yield c

        monkeypatch.setattr(client._http, "_stream", mock_stream)
        result = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        collected = []
        async for chunk in await result:
            collected.append(chunk)
        assert len(collected) == 2
        assert isinstance(collected[0], ChatCompletionChunk)
        assert collected[0].choices[0].delta.content == "Hello"
        assert collected[1].choices[0].delta.content == " world"

    @pytest.mark.asyncio
    async def test_request_headers_validated(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        request = httpx_mock.get_request()
        assert request.headers["Authorization"] == "Bearer rdk-test-key-12345"
        assert request.headers["X-RodiumAI-SDK"] == "python/0.1.0"
        assert request.headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_request_body_structure(self, httpx_mock, client, mock_chat_response):
        import json

        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        await client.chat.completions.create(
            model="pro",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["model"] == "pro"
        assert body["messages"][0]["role"] == "user"
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 100
        assert body["stream"] is False

    @pytest.mark.asyncio
    async def test_invalid_model_raises_error(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=404,
            headers={"X-Request-ID": "req-404"},
        )
        with pytest.raises(ModelNotFoundError):
            await client.chat.completions.create(
                model="unknown-model",
                messages=[{"role": "user", "content": "Hello"}],
            )
