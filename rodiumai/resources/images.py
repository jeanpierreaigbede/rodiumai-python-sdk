from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .._http import AsyncHTTPClient


@dataclass
class ImageData:
    url: Optional[str] = None
    b64_json: Optional[str] = None


@dataclass
class ImagesResponse:
    created: int = 0
    data: List[ImageData] = field(default_factory=list)


class Images:
    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def generate(
        self,
        *,
        model: str = "auto",
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        quality: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ImagesResponse:
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        if quality is not None:
            body["quality"] = quality

        status_code, data, request_id, error = await self._http._request(
            "POST", "/images/generations", json_body=body, timeout=timeout
        )
        if error:
            raise error

        images = []
        for item in data.get("data", []):
            images.append(ImageData(
                url=item.get("url"),
                b64_json=item.get("b64_json"),
            ))

        return ImagesResponse(
            created=data.get("created", 0),
            data=images,
        )
