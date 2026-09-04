from typing import Any, Dict, Optional

from .._http import AsyncHTTPClient


class Extensions:
    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def wallet(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        _, data, _, error = await self._http._request("GET", "/wallet", timeout=timeout)
        if error:
            raise error
        return data

    async def pricing(
        self,
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        params = {"model": model} if model else None
        _, data, _, error = await self._http._request(
            "GET", "/pricing", timeout=timeout, params=params
        )
        if error:
            raise error
        return data
