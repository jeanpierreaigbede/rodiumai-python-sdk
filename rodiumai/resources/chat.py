from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .._http import AsyncHTTPClient


@dataclass
class Delta:
    role: Optional[str] = None
    content: Optional[str] = None


@dataclass
class ChunkChoice:
    index: int
    delta: Delta = field(default_factory=Delta)
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionChunk:
    id: str
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[ChunkChoice] = field(default_factory=list)


@dataclass
class Message:
    role: str = ""
    content: Optional[str] = None


@dataclass
class Choice:
    index: int = 0
    message: Message = field(default_factory=Message)
    finish_reason: Optional[str] = None


@dataclass
class CompletionUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatCompletion:
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[CompletionUsage] = None
    cost_rodi: Optional[float] = None
    routing: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class Completions:
    DEFAULT_MODEL = "openai/gpt-4o"

    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str = DEFAULT_MODEL,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        if not messages:
            raise ValueError("messages must not be empty")

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }
        if temperature is not None:
            if temperature < 0 or temperature > 2:
                raise ValueError("temperature must be between 0 and 2")
            body["temperature"] = temperature
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("max_tokens must be greater than 0")
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop

        if stream:
            return self._stream_create(body, timeout)

        _, data, _, error = await self._http._request(
            "POST", "/chat/completions", json_body=body, timeout=timeout
        )
        if error:
            raise error

        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = CompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        choices_data = data.get("choices", [])
        choices = []
        for c in choices_data:
            msg = c.get("message", {})
            choices.append(
                Choice(
                    index=c.get("index", 0),
                    message=Message(
                        role=msg.get("role", ""),
                        content=msg.get("content"),
                    ),
                    finish_reason=c.get("finish_reason"),
                )
            )

        cost_rodi = data.get("cost_rodi")
        if cost_rodi is None and isinstance(data.get("rodiumai"), dict):
            cost_rodi = data["rodiumai"].get("cost_rodi")

        return ChatCompletion(
            id=data.get("id", ""),
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", model),
            choices=choices,
            usage=usage,
            cost_rodi=cost_rodi,
            routing=data.get("routing"),
            raw=data,
        )

    async def _stream_create(
        self,
        body: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk_data in self._http._stream(
            "POST", "/chat/completions", json_body=body, timeout=timeout
        ):
            choices_data = chunk_data.get("choices", [])
            if not choices_data:
                continue
            choices = []
            for c in choices_data:
                delta_data = c.get("delta", {})
                choices.append(
                    ChunkChoice(
                        index=c.get("index", 0),
                        delta=Delta(
                            role=delta_data.get("role"),
                            content=delta_data.get("content"),
                        ),
                        finish_reason=c.get("finish_reason"),
                    )
                )
            yield ChatCompletionChunk(
                id=chunk_data.get("id", ""),
                object=chunk_data.get("object", "chat.completion.chunk"),
                created=chunk_data.get("created", 0),
                model=chunk_data.get("model", ""),
                choices=choices,
            )


class ChatNamespace:
    """OpenAI-compatible namespace, also callable as flat ``client.chat(...)``."""

    def __init__(self, http_client: AsyncHTTPClient, client: Any):
        self.completions = Completions(http_client)
        self._client = client

    async def __call__(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        **options: Any,
    ) -> ChatCompletion:
        return await self._client._flat_chat(messages, **options)


class Chat:
    def __init__(self, http_client: AsyncHTTPClient, client: Any):
        self._namespace = ChatNamespace(http_client, client)
        self.completions = self._namespace.completions

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._namespace(*args, **kwargs)
