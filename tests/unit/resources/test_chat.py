import json

import pytest

from rodiumai.resources.chat import (
    ChatCompletion,
    Choice,
    Completions,
    CompletionUsage,
    Message,
    ToolCall,
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
    async def test_tool_calls_parsed_from_response(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json={
                "id": "chatcmpl-456",
                "object": "chat.completion",
                "created": 1715000000,
                "model": "auto",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Lomé"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "What's the weather in Lomé?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        message = response.choices[0].message
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id == "call_abc123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == '{"city": "Lomé"}'
        assert response.choices[0].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_no_tool_calls_defaults_to_none(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert response.choices[0].message.tool_calls is None

    @pytest.mark.asyncio
    async def test_response_message_can_be_sent_back_in_tool_loop(
        self, httpx_mock, client, mock_chat_response
    ):
        url = "https://api.rodiumai.io/v1/chat/completions"
        httpx_mock.add_response(
            url=url,
            method="POST",
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Lomé"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        httpx_mock.add_response(url=url, method="POST", json=mock_chat_response)

        user_msg = {"role": "user", "content": "What's the weather in Lomé?"}
        response = await client.chat([user_msg], tools=[{"type": "function"}])
        tool_call = response.choices[0].message.tool_calls[0]
        follow_up = await client.chat(
            [
                user_msg,
                response.choices[0].message,
                {"role": "tool", "tool_call_id": tool_call.id, "content": '{"temp_c": 32}'},
            ]
        )

        assert follow_up.choices[0].message.content == "Hello! How can I help you?"
        sent = json.loads(httpx_mock.get_request().content)["messages"]
        assert sent == [
            user_msg,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Lomé"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": '{"temp_c": 32}'},
        ]

    def test_message_to_dict_without_tool_calls(self):
        assert Message(role="assistant", content="hi").to_dict() == {
            "role": "assistant",
            "content": "hi",
        }

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
    async def test_streaming_tool_call_deltas_assembled_correctly(self, monkeypatch, client):
        chunks = [
            {
                "id": "chunk-1",
                "object": "chat.completion.chunk",
                "created": 100,
                "model": "auto",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chunk-1",
                "object": "chat.completion.chunk",
                "created": 100,
                "model": "auto",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"city": "Lomé"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]

        async def mock_stream(*args, **kwargs):
            for c in chunks:
                yield c

        monkeypatch.setattr(client._http, "_stream", mock_stream)
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "What's the weather in Lomé?"}],
            stream=True,
        )
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        assert len(collected) == 2
        first_call = collected[0].choices[0].delta.tool_calls[0]
        assert first_call.id == "call_abc123"
        assert first_call.function.name == "get_weather"
        second_call = collected[1].choices[0].delta.tool_calls[0]
        assert second_call.function.arguments == '{"city": "Lomé"}'
        assert collected[1].choices[0].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_empty_message_list_raises_value_error(self):
        completions = Completions(None)
        with pytest.raises(ValueError, match="messages must not be empty"):
            await completions.create(messages=[])

    @pytest.mark.asyncio
    async def test_temperature_out_of_range_raises_value_error(self):
        completions = Completions(None)
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            await completions.create(
                messages=[{"role": "user", "content": "hi"}],
                temperature=3.0,
            )

    @pytest.mark.asyncio
    async def test_temperature_below_zero_raises_value_error(self):
        completions = Completions(None)
        with pytest.raises(ValueError):
            await completions.create(
                messages=[{"role": "user", "content": "hi"}],
                temperature=-1.0,
            )
