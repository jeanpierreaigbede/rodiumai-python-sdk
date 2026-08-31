from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .._http import AsyncHTTPClient


@dataclass
class VideoData:
    url: Optional[str] = None
    b64_json: Optional[str] = None


@dataclass
class VideoResponse:
    created: int = 0
    data: List[VideoData] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class Generations:
    DEFAULT_MODEL = "openai/gpt-4o"

    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str = DEFAULT_MODEL,
        prompt: str,
        duration_seconds: Optional[int] = None,
        timeout: Optional[float] = 600.0,
        **kwargs: Any,
    ) -> VideoResponse:
        body: Dict[str, Any] = {"model": model, "prompt": prompt, **kwargs}
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds

        _, data, _, error = await self._http._request(
            "POST", "/videos/generations", json_body=body, timeout=timeout
        )
        if error:
            raise error

        items = [
            VideoData(url=item.get("url"), b64_json=item.get("b64_json"))
            for item in data.get("data", [])
        ]
        return VideoResponse(created=data.get("created", 0), data=items, raw=data)


class VideoNamespace:
    def __init__(self, http_client: AsyncHTTPClient, client: Any):
        self.generations = Generations(http_client)
        self._client = client

    async def __call__(self, **options: Any) -> VideoResponse:
        model = options.pop("model", self._client._resolve_model())
        timeout = options.pop("timeout", max(self._client._timeout, 600.0))
        return await self.generations.create(model=model, timeout=timeout, **options)


class Video:
    def __init__(self, http_client: AsyncHTTPClient, client: Any):
        self._namespace = VideoNamespace(http_client, client)
        self.generations = self._namespace.generations

    async def __call__(self, **options: Any) -> VideoResponse:
        return await self._namespace(**options)
