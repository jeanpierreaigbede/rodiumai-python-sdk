import pytest

from rodiumai.errors import ModelNotFoundError
from rodiumai.resources.chat import (
    Chat,
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    CompletionUsage,
    Message,
)


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_standard_response_parsed_correctly(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(response, ChatCompletion)
        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion"
        assert response.created == 1715000000
        assert response.model == "auto"
        assert isinstance(response.choices[0], Choice)
        assert isinstance(response.choices[0].message, Message)
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.choices[0].finish_reason == "stop"
        assert isinstance(response.usage, CompletionUsage)
        assert response.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_streaming_chunks_assembled_correctly(self, monkeypatch, client):
        chunks = [
            {
                "id": "chunk-1",
                "object": "chat.completion.chunk",
                "created": 100,
                "model": "auto",
                "choices": [{"index": 0, "delta": {"content": "A"}, "finish_reason": None}],
            },
            {
                "id": "chunk-1",
                "object": "chat.completion.chunk",
                "created": 100,
                "model": "auto",
                "choices": [{"index": 0, "delta": {"content": "B"}, "finish_reason": None}],
            },
        ]

        async def mock_stream(*args, **kwargs):
            for c in chunks:
                yield c

        monkeypatch.setattr(client._http, "_stream", mock_stream)
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert len(collected) == 2
        assert collected[0].choices[0].delta.content == "A"
        assert collected[1].choices[0].delta.content == "B"
        assert collected[0].object == "chat.completion.chunk"

    @pytest.mark.asyncio
    async def test_empty_message_list_raises_value_error(self):
        chat = Chat(None)
        with pytest.raises(ValueError, match="messages must not be empty"):
            await chat.completions.create(messages=[])

    @pytest.mark.asyncio
    async def test_temperature_out_of_range_raises_value_error(self):
        chat = Chat(None)
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            await chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                temperature=3.0,
            )

    @pytest.mark.asyncio
    async def test_temperature_below_zero_raises_value_error(self):
        chat = Chat(None)
        with pytest.raises(ValueError):
            await chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                temperature=-1.0,
            )
