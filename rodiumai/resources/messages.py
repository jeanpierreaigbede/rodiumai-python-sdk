from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .._http import AsyncHTTPClient

_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class MessageContentBlock:
    type: str = "text"
    text: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MessageResponse:
    """Anthropic-shaped ``/v1/messages`` result (non-streaming)."""

    id: str = ""
    type: str = "message"
    role: str = "assistant"
    model: str = ""
    content: List[MessageContentBlock] = field(default_factory=list)
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Optional[MessageUsage] = None
    cost_rodi: Optional[float] = None
    routing: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Convenience aggregation of every ``text`` block."""
        return "".join(b.text for b in self.content if b.type == "text" and b.text)


@dataclass
class MessageStreamEvent:
    """A single Anthropic streaming SSE event.

    The Anthropic protocol emits named events (``message_start``,
    ``content_block_delta`` carrying ``delta.text``, ``message_stop`` …) and,
    unlike the OpenAI chat stream, has **no** ``[DONE]`` sentinel — iteration
    ends when the upstream closes the stream.
    """

    type: str = ""
    index: Optional[int] = None
    delta: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _build_message_response(data: Dict[str, Any], fallback_model: str) -> MessageResponse:
    content = [
        MessageContentBlock(
            type=block.get("type", "text"),
            text=block.get("text"),
            raw=block,
        )
        for block in data.get("content", [])
    ]

    usage_data = data.get("usage")
    usage = None
    if isinstance(usage_data, dict):
        usage = MessageUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
        )

    cost_rodi = data.get("cost_rodi")
    if cost_rodi is None and isinstance(data.get("rodiumai"), dict):
        cost_rodi = data["rodiumai"].get("cost_rodi")

    return MessageResponse(
        id=data.get("id", ""),
        type=data.get("type", "message"),
        role=data.get("role", "assistant"),
        model=data.get("model", fallback_model),
        content=content,
        stop_reason=data.get("stop_reason"),
        stop_sequence=data.get("stop_sequence"),
        usage=usage,
        cost_rodi=cost_rodi,
        routing=data.get("routing"),
        raw=data,
    )


class Messages:
    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[MessageResponse, AsyncIterator[MessageStreamEvent]]:
        anthropic_version = kwargs.pop("anthropic_version", _DEFAULT_ANTHROPIC_VERSION)
        headers = {
            "x-api-key": self._http._api_key,
            "anthropic-version": anthropic_version,
        }
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }

        if stream:
            body["stream"] = True
            return self._stream_create(body, timeout, headers)

        body["stream"] = False
        _, data, _, error = await self._http._request(
            "POST",
            "/messages",
            json_body=body,
            timeout=timeout,
            extra_headers=headers,
        )
        if error:
            raise error
        return _build_message_response(data, model)

    async def _stream_create(
        self,
        body: Dict[str, Any],
        timeout: Optional[float],
        headers: Dict[str, str],
    ) -> AsyncIterator[MessageStreamEvent]:
        async for event_data in self._http._stream(
            "POST",
            "/messages",
            json_body=body,
            timeout=timeout,
            extra_headers=headers,
        ):
            yield MessageStreamEvent(
                type=event_data.get("type", ""),
                index=event_data.get("index"),
                delta=event_data.get("delta"),
                raw=event_data,
            )


class MessagesNamespace:
    def __init__(self, http_client: AsyncHTTPClient):
        self._messages = Messages(http_client)

    async def create(
        self, **options: Any
    ) -> Union[MessageResponse, AsyncIterator[MessageStreamEvent]]:
        return await self._messages.create(**options)

    async def __call__(
        self, **options: Any
    ) -> Union[MessageResponse, AsyncIterator[MessageStreamEvent]]:
        return await self.create(**options)
