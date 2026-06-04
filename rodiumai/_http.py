import asyncio
import random
import uuid
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx

from ._version import VERSION
from .errors import (
    NetworkError,
    RateLimitError,
    RodiumAIError,
    TimeoutError_,
    map_http_status,
)


class AsyncHTTPClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.rodiumai.io/v1",
        timeout: float = 30.0,
        stream_timeout: float = 600.0,
        max_retries: int = 3,
    ):
        if base_url.startswith("http://"):
            raise ValueError(
                "HTTPS is required. "
                "Use 'https://' URL scheme. "
                "See: https://docs.rodiumai.io/security"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._stream_timeout = stream_timeout
        self._max_retries = max_retries

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
        )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "X-RodiumAI-SDK": f"python/{VERSION}",
            "X-RodiumAI-Version": VERSION,
            "Content-Type": "application/json",
            "User-Agent": f"RodiumAI-Python-SDK/{VERSION}",
        }

    def _get_request_id(self) -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _mask_key(key: str) -> str:
        if len(key) <= 8:
            return "****"
        return f"{key[:4]}****{key[-4:]}"

    async def _request(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
    ) -> Tuple[int, Dict[str, Any], Optional[str], Optional[RodiumAIError]]:
        url = f"{self._base_url}{path}"
        headers = self._get_headers()
        request_id = self._get_request_id()
        effective_timeout = timeout if timeout is not None else self._timeout
        retry_count = 0
        last_error: Optional[RodiumAIError] = None

        while retry_count <= self._max_retries:
            try:
                async with self._client.stream(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    files=files,
                    timeout=effective_timeout,
                ) as response:
                    resp_request_id = response.headers.get("X-Request-ID", request_id)
                    body = await response.aread()
                    data: Dict[str, Any] = {}
                    if body:
                        import json as _json

                        data = _json.loads(body) if body else {}

                    if response.status_code < 400:
                        return response.status_code, data, resp_request_id, None

                    error = map_http_status(response.status_code, resp_request_id)
                    backend_msg = None
                    if isinstance(data, dict):
                        err_data = data.get("error")
                        if isinstance(err_data, dict):
                            backend_msg = err_data.get("message")
                        elif isinstance(err_data, str):
                            backend_msg = err_data
                    if backend_msg:
                        error.message = f"{error.message} - {backend_msg}"

                    if response.status_code == 429:
                        retry_after_str = response.headers.get("Retry-After", "1")
                        try:
                            retry_after = float(retry_after_str)
                        except ValueError:
                            retry_after = 1.0
                        if isinstance(error, RateLimitError):
                            error.retry_after = retry_after

                    if response.status_code in (429, 500, 502, 503, 504):
                        if retry_count < self._max_retries:
                            retry_count += 1
                            sleep_time = (2 ** (retry_count - 1)) * random.uniform(0.9, 1.1)
                            await asyncio.sleep(sleep_time)
                            continue

                    return response.status_code, data, resp_request_id, error

            except httpx.TimeoutException as e:
                retry_count += 1
                if retry_count <= self._max_retries:
                    sleep_time = (2 ** (retry_count - 1)) * random.uniform(0.9, 1.1)
                    await asyncio.sleep(sleep_time)
                    continue
                raise TimeoutError_(elapsed=effective_timeout) from e

            except httpx.NetworkError as e:
                retry_count += 1
                if retry_count <= self._max_retries:
                    sleep_time = (2 ** (retry_count - 1)) * random.uniform(0.9, 1.1)
                    await asyncio.sleep(sleep_time)
                    continue
                raise NetworkError(str(e)) from e

        raise last_error or RodiumAIError("Request failed after retries")

    async def _stream(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        url = f"{self._base_url}{path}"
        headers = self._get_headers()
        effective_timeout = timeout if timeout is not None else self._stream_timeout

        async with httpx.AsyncClient(timeout=httpx.Timeout(effective_timeout)) as client:
            async with client.stream(
                method,
                url,
                headers=headers,
                json=json_body,
            ) as response:
                request_id = response.headers.get("X-Request-ID", self._get_request_id())

                if response.status_code >= 400:
                    error = map_http_status(response.status_code, request_id)
                    raise error

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            return
                        if payload:
                            import json as _json

                            yield _json.loads(payload)

    def _get_usage_from_response(self, data: Dict[str, Any]) -> Optional[Dict[str, int]]:
        usage = data.get("usage")
        if usage and isinstance(usage, dict):
            return {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            }
        return None
