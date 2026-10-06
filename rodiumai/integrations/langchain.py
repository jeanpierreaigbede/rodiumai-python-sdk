"""LangChain-compatible chat model backed by the RodiumAI SDK.

Lets a :class:`rodiumai.RodiumAI` client be used as a drop-in model in
LangChain / LangGraph agents, including tool-calling agents built with
``bind_tools`` (e.g. ``langgraph.prebuilt.create_react_agent``).

Requires the optional ``langchain-core`` dependency::

    pip install rodiumai[langchain]

Example::

    from rodiumai import RodiumAI
    from rodiumai.integrations.langchain import ChatRodiumAI
    from langgraph.prebuilt import create_react_agent

    llm = ChatRodiumAI(client=RodiumAI(api_key="..."), model="openai/gpt-4o")
    agent = create_react_agent(llm, tools=[my_tool])

The RodiumAI SDK is async-only, so this adapter only implements the async
LangChain entry points (``ainvoke``, ``astream`` is not yet supported).
Use ``await agent.ainvoke(...)`` rather than the sync ``.invoke(...)``.
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import ConfigDict
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "rodiumai.integrations.langchain requires the optional 'langchain-core' "
        "dependency. Install it with: pip install rodiumai[langchain]"
    ) from exc

from ..client import RodiumAI
from ..resources.chat import Message as RodiumAIMessage

DEFAULT_MODEL = "openai/gpt-4o"


def _lc_message_to_rodiumai(message: BaseMessage) -> Dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    if isinstance(message, AIMessage):
        payload: Dict[str, Any] = {"role": "assistant", "content": message.content or None}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["args"]),
                    },
                }
                for tool_call in message.tool_calls
            ]
        return payload
    role = getattr(message, "role", None) or message.type
    return {"role": role, "content": message.content}


def _rodiumai_message_to_ai_message(message: RodiumAIMessage) -> AIMessage:
    tool_calls = []
    invalid_tool_calls = []
    for tool_call in message.tool_calls or []:
        raw_args = tool_call.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            args = None
        if not isinstance(args, dict):
            invalid_tool_calls.append(
                {
                    "name": tool_call.function.name,
                    "args": raw_args,
                    "id": tool_call.id,
                    "error": "Model returned tool arguments that are not a JSON object.",
                    "type": "invalid_tool_call",
                }
            )
            continue
        tool_calls.append(
            {
                "name": tool_call.function.name,
                "args": args,
                "id": tool_call.id,
                "type": "tool_call",
            }
        )
    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )


class ChatRodiumAI(BaseChatModel):
    """LangChain ``BaseChatModel`` wrapping a :class:`rodiumai.RodiumAI` client.

    Async-only (``ainvoke``/``agenerate``), matching the underlying SDK.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: RodiumAI
    model: str = DEFAULT_MODEL
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    @property
    def _llm_type(self) -> str:
        return "rodiumai-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model}

    def _request_kwargs(self, stop: Optional[List[str]], extra: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": extra.pop("model", self.model)}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if stop:
            kwargs["stop"] = stop
        kwargs.update(extra)
        return kwargs

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "ChatRodiumAI is async-only because the RodiumAI SDK client is async-only. "
            "Use `await model.ainvoke(...)` / `await agent.ainvoke(...)` instead of the "
            "sync `.invoke(...)`."
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        rodiumai_messages = [_lc_message_to_rodiumai(m) for m in messages]
        request_kwargs = self._request_kwargs(stop, dict(kwargs))
        response = await self.client.chat.completions.create(
            messages=rodiumai_messages, **request_kwargs
        )
        choice = response.choices[0]
        ai_message = _rodiumai_message_to_ai_message(choice.message)
        generation = ChatGeneration(
            message=ai_message,
            generation_info={"finish_reason": choice.finish_reason},
        )
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Any]],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)
