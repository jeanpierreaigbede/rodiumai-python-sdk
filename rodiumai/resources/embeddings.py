from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .._http import AsyncHTTPClient


@dataclass
class Embedding:
    object: str = "embedding"
    index: int = 0
    embedding: List[float] = field(default_factory=list)


@dataclass
class EmbeddingUsage:
    prompt_tokens: int = 0
    total_tokens: int = 0


@dataclass
class EmbeddingsResponse:
    object: str = "list"
    data: List[Embedding] = field(default_factory=list)
    model: str = ""
    usage: Optional[EmbeddingUsage] = None


class Embeddings:
    DEFAULT_MODEL = "openai/gpt-4o"

    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str = DEFAULT_MODEL,
        input: Union[str, List[str]],
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> EmbeddingsResponse:
        body: Dict[str, Any] = {
            "model": model,
            "input": input,
            **kwargs,
        }

        _, data, _, error = await self._http._request(
            "POST", "/embeddings", json_body=body, timeout=timeout
        )
        if error:
            raise error

        embeddings_list = []
        for item in data.get("data", []):
            embeddings_list.append(
                Embedding(
                    object=item.get("object", "embedding"),
                    index=item.get("index", 0),
                    embedding=item.get("embedding", []),
                )
            )

        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = EmbeddingUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return EmbeddingsResponse(
            object=data.get("object", "list"),
            data=embeddings_list,
            model=data.get("model", model),
            usage=usage,
        )


class EmbeddingsNamespace:
    def __init__(self, http_client: AsyncHTTPClient, client: Any):
        self._embeddings = Embeddings(http_client)
        self._client = client

    async def create(self, **kwargs: Any) -> EmbeddingsResponse:
        return await self._embeddings.create(**kwargs)

    async def __call__(
        self,
        input: Union[str, List[str]],
        **options: Any,
    ) -> EmbeddingsResponse:
        model = options.pop("model", self._client._resolve_model())
        timeout = options.pop("timeout", None)
        return await self.create(model=model, input=input, timeout=timeout, **options)
