from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .._http import AsyncHTTPClient


@dataclass
class ResponseOutputContent:
    type: str = ""
    text: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseOutputItem:
    type: str = ""
    role: Optional[str] = None
    content: List[ResponseOutputContent] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponsesUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ResponsesResponse:
    """OpenAI-shaped ``/v1/responses`` result (non-streaming)."""

    id: str = ""
    object: str = "response"
    created_at: Optional[int] = None
    model: str = ""
    status: Optional[str] = None
    output: List[ResponseOutputItem] = field(default_factory=list)
    output_text: str = ""
    usage: Optional[ResponsesUsage] = None
    cost_rodi: Optional[float] = None
    routing: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseStreamEvent:
    """A single Responses streaming SSE event.

    Most notably ``response.output_text.delta`` (carrying a ``delta`` string).
    There is no ``[DONE]`` sentinel — iteration ends when the upstream closes
    the stream.
    """

    type: str = ""
    delta: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _aggregate_output_text(output: List[ResponseOutputItem]) -> str:
    parts: List[str] = []
    for item in output:
        for block in item.content:
            if block.type == "output_text" and block.text:
                parts.append(block.text)
    return "".join(parts)


def _build_responses_response(data: Dict[str, Any], fallback_model: str) -> ResponsesResponse:
    output = [
        ResponseOutputItem(
            type=item.get("type", ""),
            role=item.get("role"),
            content=[
                ResponseOutputContent(
                    type=block.get("type", ""),
                    text=block.get("text"),
                    raw=block,
                )
                for block in item.get("content", [])
            ],
            raw=item,
        )
        for item in data.get("output", [])
    ]

    output_text = data.get("output_text")
    if not isinstance(output_text, str):
        output_text = _aggregate_output_text(output)

    usage_data = data.get("usage")
    usage = None
    if isinstance(usage_data, dict):
        usage = ResponsesUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

    cost_rodi = data.get("cost_rodi")
    if cost_rodi is None and isinstance(data.get("rodiumai"), dict):
        cost_rodi = data["rodiumai"].get("cost_rodi")

    return ResponsesResponse(
        id=data.get("id", ""),
        object=data.get("object", "response"),
        created_at=data.get("created_at"),
        model=data.get("model", fallback_model),
        status=data.get("status"),
        output=output,
        output_text=output_text,
        usage=usage,
        cost_rodi=cost_rodi,
        routing=data.get("routing"),
        raw=data,
    )


class Responses:
    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str,
        stream: bool = False,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[ResponsesResponse, AsyncIterator[ResponseStreamEvent]]:
        if not model:
            raise ValueError("model is required for responses.create")

        body: Dict[str, Any] = {"model": model, **kwargs}

        if stream:
            body["stream"] = True
            return self._stream_create(body, timeout)

        body["stream"] = False
        _, data, _, error = await self._http._request(
            "POST", "/responses", json_body=body, timeout=timeout
        )
        if error:
            raise error
        return _build_responses_response(data, model)

    async def _stream_create(
        self,
        body: Dict[str, Any],
        timeout: Optional[float],
    ) -> AsyncIterator[ResponseStreamEvent]:
        async for event_data in self._http._stream(
            "POST", "/responses", json_body=body, timeout=timeout
        ):
            yield ResponseStreamEvent(
                type=event_data.get("type", ""),
                delta=event_data.get("delta"),
                raw=event_data,
            )


class ResponsesNamespace:
    def __init__(self, http_client: AsyncHTTPClient):
        self._responses = Responses(http_client)

    async def create(
        self, **options: Any
    ) -> Union[ResponsesResponse, AsyncIterator[ResponseStreamEvent]]:
        return await self._responses.create(**options)

    async def __call__(
        self, **options: Any
    ) -> Union[ResponsesResponse, AsyncIterator[ResponseStreamEvent]]:
        return await self.create(**options)
