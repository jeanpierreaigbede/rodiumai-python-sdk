import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from rodiumai.integrations.langchain import (  # noqa: E402
    ChatRodiumAI,
    _lc_message_to_rodiumai,
    _rodiumai_message_to_ai_message,
)
from rodiumai.resources.chat import (  # noqa: E402
    ChatCompletion,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
)


class TestMessageConversion:
    def test_human_message(self):
        assert _lc_message_to_rodiumai(HumanMessage(content="hi")) == {
            "role": "user",
            "content": "hi",
        }

    def test_system_message(self):
        assert _lc_message_to_rodiumai(SystemMessage(content="be nice")) == {
            "role": "system",
            "content": "be nice",
        }

    def test_tool_message(self):
        result = _lc_message_to_rodiumai(
            ToolMessage(content='{"temp_c": 32}', tool_call_id="call_1")
        )
        assert result == {
            "role": "tool",
            "content": '{"temp_c": 32}',
            "tool_call_id": "call_1",
        }

    def test_ai_message_without_tool_calls(self):
        result = _lc_message_to_rodiumai(AIMessage(content="Hello!"))
        assert result == {"role": "assistant", "content": "Hello!"}

    def test_ai_message_with_tool_calls(self):
        ai_message = AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": "Lomé"}, "id": "call_1"}],
        )
        result = _lc_message_to_rodiumai(ai_message)
        assert result["role"] == "assistant"
        assert result["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lomé"})},
            }
        ]

    def test_generic_chat_message_fallback(self):
        result = _lc_message_to_rodiumai(ChatMessage(content="hey", role="user"))
        assert result == {"role": "user", "content": "hey"}

    def test_rodiumai_message_without_tool_calls(self):
        ai_message = _rodiumai_message_to_ai_message(Message(role="assistant", content="Hi!"))
        assert ai_message.content == "Hi!"
        assert ai_message.tool_calls == []
        assert ai_message.invalid_tool_calls == []

    def test_rodiumai_message_with_valid_tool_calls(self):
        message = Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="get_weather", arguments='{"city": "Lomé"}'),
                )
            ],
        )
        ai_message = _rodiumai_message_to_ai_message(message)
        assert ai_message.tool_calls == [
            {"name": "get_weather", "args": {"city": "Lomé"}, "id": "call_1", "type": "tool_call"}
        ]
        assert ai_message.invalid_tool_calls == []

    def test_rodiumai_message_with_invalid_json_tool_call_arguments(self):
        message = Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="get_weather", arguments="not json"),
                )
            ],
        )
        ai_message = _rodiumai_message_to_ai_message(message)
        assert ai_message.tool_calls == []
        assert len(ai_message.invalid_tool_calls) == 1
        assert ai_message.invalid_tool_calls[0]["name"] == "get_weather"
        assert ai_message.invalid_tool_calls[0]["args"] == "not json"


class TestChatRodiumAI:
    def _make_llm(self, client, **overrides):
        return ChatRodiumAI(client=client, model="openai/gpt-4o", **overrides)

    def test_llm_type_and_identifying_params(self, client):
        llm = self._make_llm(client)
        assert llm._llm_type == "rodiumai-chat"
        assert llm._identifying_params == {"model": "openai/gpt-4o"}

    def test_sync_generate_not_implemented(self, client):
        llm = self._make_llm(client)
        with pytest.raises(NotImplementedError):
            llm.invoke([HumanMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_agenerate_without_tool_calls(self, client, monkeypatch):
        captured = {}

        async def fake_create(*, model, messages, **kwargs):
            captured["model"] = model
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return ChatCompletion(
                id="chatcmpl-1",
                model=model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content="Hello!"),
                        finish_reason="stop",
                    )
                ],
            )

        monkeypatch.setattr(client.chat.completions, "create", fake_create)
        llm = self._make_llm(client, temperature=0.2, max_tokens=100, top_p=0.9)

        result = await llm.ainvoke(
            [SystemMessage(content="be nice"), HumanMessage(content="hi")], stop=["END"]
        )

        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"
        assert result.response_metadata["finish_reason"] == "stop"
        assert captured["model"] == "openai/gpt-4o"
        assert captured["messages"] == [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]
        assert captured["kwargs"]["temperature"] == 0.2
        assert captured["kwargs"]["max_tokens"] == 100
        assert captured["kwargs"]["top_p"] == 0.9
        assert captured["kwargs"]["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_agenerate_with_tool_calls(self, client, monkeypatch):
        async def fake_create(*, model, messages, **kwargs):
            assert kwargs["tools"][0]["function"]["name"] == "get_weather"
            return ChatCompletion(
                id="chatcmpl-1",
                model=model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    type="function",
                                    function=FunctionCall(
                                        name="get_weather",
                                        arguments=json.dumps({"city": "Lomé"}),
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
            )

        monkeypatch.setattr(client.chat.completions, "create", fake_create)
        llm = self._make_llm(client)
        llm_with_tools = llm.bind_tools(
            [
                {
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
                }
            ]
        )

        result = await llm_with_tools.ainvoke([HumanMessage(content="Weather in Lomé?")])

        assert result.tool_calls == [
            {"name": "get_weather", "args": {"city": "Lomé"}, "id": "call_1", "type": "tool_call"}
        ]

    def test_bind_tools_passes_tool_choice(self, client):
        llm = self._make_llm(client)
        bound = llm.bind_tools(
            [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object"}},
                }
            ],
            tool_choice="auto",
        )
        assert bound.kwargs["tool_choice"] == "auto"
        assert bound.kwargs["tools"][0]["function"]["name"] == "get_weather"
