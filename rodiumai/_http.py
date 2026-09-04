import asyncio
import json as _json
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

DEFAULT_BASE_URL = "https://api.rodiumai.io/v1"


def _extract_error_info(data: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse OpenAI-shaped or Anthropic-shaped error envelopes."""
    if not isinstance(data, dict):
        return None, None

    if data.get("type") == "error" and isinstance(data.get("error"), dict):
        err = data["error"]
        return err.get("message"), err.get("type")

    err = data.get("error")
    if isinstance(err, dict):
        return err.get("message"), err.get("code") or err.get("type")
    if isinstance(err, str):
        return err, None
    return None, None


class AsyncHTTPClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        stream_timeout: float = 600.0,
        max_retries: int = 3,
    ):
        if base_url.startswith("http://") and "localhost" not in base_url:
            raise ValueError(
                "HTTPS is required for production. "
                "Use http://localhost:8001/v1 for local development only."
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

    def _get_headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-RodiumAI-SDK": f"python/{VERSION}",
            "X-RodiumAI-Version": VERSION,
            "User-Agent": f"RodiumAI-Python-SDK/{VERSION}",
        }
        if extra:
            headers.update(extra)
        return headers

    def _get_request_id(self) -> str:
        return str(uuid.uuid4())

    async def _request(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, Any], Optional[str], Optional[RodiumAIError]]:
        url = path
        headers = self._get_headers(extra_headers)
        request_id = self._get_request_id()
        effective_timeout = timeout if timeout is not None else self._timeout
        retry_count = 0

        while retry_count <= self._max_retries:
            try:
                request_kwargs: Dict[str, Any] = {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "timeout": effective_timeout,
                    "params": params,
                }
                if files is not None:
                    request_kwargs["files"] = files
                    request_kwargs["data"] = data or {}
                    headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
                    request_kwargs["headers"] = headers
                elif json_body is not None:
                    request_kwargs["json"] = json_body
                    headers.setdefault("Content-Type", "application/json")
                    request_kwargs["headers"] = headers

                response = await self._client.request(**request_kwargs)
                resp_request_id = response.headers.get("X-Request-ID", request_id)
                body = response.content
                data_out: Dict[str, Any] = {}
                if body and response.headers.get("content-type", "").startswith("application/json"):
                    data_out = _json.loads(body)

                if response.status_code < 400:
                    return response.status_code, data_out, resp_request_id, None

                error = self._map_response_error(
                    response.status_code,
                    data_out,
                    resp_request_id,
                    response.headers,
                )

                if response.status_code in (429, 500, 502, 503, 504):
                    if retry_count < self._max_retries:
                        retry_count += 1
                        sleep_time = (2 ** (retry_count - 1)) * random.uniform(0.9, 1.1)
                        await asyncio.sleep(sleep_time)
                        continue

                return response.status_code, data_out, resp_request_id, error

            except httpx.TimeoutException as e:
                retry_count += 1
                if retry_count <= self._max_retries:
                    await asyncio.sleep((2 ** (retry_count - 1)) * random.uniform(0.9, 1.1))
                    continue
                raise TimeoutError_(elapsed=effective_timeout) from e

            except httpx.NetworkError as e:
                retry_count += 1
                if retry_count <= self._max_retries:
                    await asyncio.sleep((2 ** (retry_count - 1)) * random.uniform(0.9, 1.1))
                    continue
                raise NetworkError(str(e)) from e

        raise RodiumAIError("Request failed after retries")  # pragma: no cover

    @staticmethod
    def _mask_key(key: str) -> str:
        if len(key) <= 4:
            return "****"
        if key.startswith("rdk-") and len(key) > 12:
            return f"rdk-****{key[-4:]}"
        if key.startswith("rd_sk_") and len(key) > 12:
            return f"rd_sk_****{key[-4:]}"
        return f"{key[:4]}****{key[-4:]}"

    @staticmethod
    def _get_usage_from_response(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if prompt is None and completion is None and total is None:
            return None
        return {
            "prompt": int(prompt or 0),
            "completion": int(completion or 0),
            "total": int(total or 0),
        }

    async def _request_binary(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[bytes, str, Optional[RodiumAIError]]:
        headers = self._get_headers({"Content-Type": "application/json"})
        effective_timeout = timeout if timeout is not None else self._timeout

        response = await self._client.request(
            method=method,
            url=path,
            headers=headers,
            json=json_body,
            timeout=effective_timeout,
        )

        if response.status_code < 400:
            content_type = response.headers.get("content-type", "application/octet-stream")
            return response.content, content_type, None

        data_out: Dict[str, Any] = {}
        if response.content and "json" in response.headers.get("content-type", ""):
            data_out = _json.loads(response.content)

        request_id = response.headers.get("X-Request-ID")
        error = self._map_response_error(
            response.status_code,
            data_out,
            request_id,
            response.headers,
        )
        return b"", "", error

    def _map_response_error(
        self,
        status_code: int,
        data: Dict[str, Any],
        request_id: Optional[str],
        headers: httpx.Headers,
    ) -> RodiumAIError:
        error = map_http_status(status_code, request_id)
        backend_msg, backend_code = _extract_error_info(data)
        if backend_msg:
            error.message = backend_msg
        if backend_code:
            error.error_code = backend_code

        if status_code == 429:
            retry_after_str = headers.get("Retry-After", "1")
            try:
                retry_after = float(retry_after_str)
            except ValueError:
                retry_after = 1.0
            if isinstance(error, RateLimitError):
                error.retry_after = retry_after

        return error

    async def _stream(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        headers = self._get_headers({"Content-Type": "application/json"})
        effective_timeout = timeout if timeout is not None else self._stream_timeout

        async with httpx.AsyncClient(timeout=httpx.Timeout(effective_timeout)) as client:
            async with client.stream(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                json=json_body,
            ) as response:
                request_id = response.headers.get("X-Request-ID", self._get_request_id())

                if response.status_code >= 400:
                    body = await response.aread()
                    data: Dict[str, Any] = _json.loads(body) if body else {}
                    raise self._map_response_error(
                        response.status_code,
                        data,
                        request_id,
                        response.headers,
                    )

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            return
                        if payload:
                            yield _json.loads(payload)
