from typing import Any, Dict, List, Optional

from .._http import AsyncHTTPClient


class Messages:
    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: List[Dict[str, Any]],
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        anthropic_version = kwargs.pop("anthropic_version", "2023-06-01")
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }
        headers = {
            "x-api-key": self._http._api_key,
            "anthropic-version": anthropic_version,
        }
        _, data, _, error = await self._http._request(
            "POST",
            "/messages",
            json_body=body,
            timeout=timeout,
            extra_headers=headers,
        )
        if error:
            raise error
        return data


class MessagesNamespace:
    def __init__(self, http_client: AsyncHTTPClient):
        self._messages = Messages(http_client)

    async def create(self, **options: Any) -> Dict[str, Any]:
        return await self._messages.create(**options)

    async def __call__(self, **options: Any) -> Dict[str, Any]:
        return await self.create(**options)
