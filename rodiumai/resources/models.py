from typing import Any, Dict, Optional

from .._http import AsyncHTTPClient


class Models:
    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def list(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        _, data, _, error = await self._http._request("GET", "/models", timeout=timeout)
        if error:
            raise error
        return data

    async def retrieve(self, model_id: str, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        _, data, _, error = await self._http._request(
            "GET", f"/models/{model_id}", timeout=timeout
        )
        if error:
            raise error
        return data

    async def list_coding(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        _, data, _, error = await self._http._request("GET", "/models/coding", timeout=timeout)
        if error:
            raise error
        return data


class ModelsNamespace:
    """Flat ``client.models()`` and OpenAI-style ``client.models.list()``."""

    def __init__(self, http_client: AsyncHTTPClient):
        self._models = Models(http_client)

    async def __call__(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        return await self.list(timeout=timeout)

    async def list(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        return await self._models.list(timeout=timeout)

    async def retrieve(self, model_id: str, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        return await self._models.retrieve(model_id, timeout=timeout)

    async def list_coding(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        return await self._models.list_coding(timeout=timeout)
