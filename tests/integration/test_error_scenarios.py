import json

import pytest

from rodiumai.errors import (
    InsufficientRODIError,
    InternalServerError,
    InvalidAPIKeyError,
    RateLimitError,
    RodiumAIError,
    ServiceUnavailableError,
    TimeoutError_,
)


class TestErrorScenarios:
    @pytest.mark.asyncio
    async def test_server_401_raises_invalid_api_key(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=401,
            headers={"X-Request-ID": "req-401"},
        )
        with pytest.raises(InvalidAPIKeyError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_server_402_raises_insufficient_rodi(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=402,
            headers={"X-Request-ID": "req-402"},
        )
        with pytest.raises(InsufficientRODIError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_server_429_with_retry_after_raises_rate_limit(self, httpx_mock, client):
        for _ in range(4):
            httpx_mock.add_response(
                url="https://api.rodiumai.io/v1/chat/completions",
                method="POST",
                status_code=429,
                headers={"X-Request-ID": "req-429", "Retry-After": "2.5"},
            )
        with pytest.raises(RateLimitError) as exc_info:
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert exc_info.value.retry_after == 2.5

    @pytest.mark.asyncio
    async def test_server_429_with_retry_after_float(self, httpx_mock, client):
        for _ in range(4):
            httpx_mock.add_response(
                url="https://api.rodiumai.io/v1/chat/completions",
                method="POST",
                status_code=429,
                headers={"X-Request-ID": "req-429", "Retry-After": "2"},
            )
        with pytest.raises(RateLimitError) as exc_info:
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert exc_info.value.retry_after == 2.0

    @pytest.mark.asyncio
    async def test_server_500_retried_3_times(self, httpx_mock, client):
        for _ in range(4):
            httpx_mock.add_response(
                url="https://api.rodiumai.io/v1/chat/completions",
                method="POST",
                status_code=500,
                headers={"X-Request-ID": "req-500"},
            )
        with pytest.raises(InternalServerError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_server_503_retried_3_times(self, httpx_mock, client):
        for _ in range(4):
            httpx_mock.add_response(
                url="https://api.rodiumai.io/v1/chat/completions",
                method="POST",
                status_code=503,
                headers={"X-Request-ID": "req-503"},
            )
        with pytest.raises(ServiceUnavailableError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_server_slow_raises_timeout_error(self, httpx_mock, client):
        import httpx
        for _ in range(4):
            httpx_mock.add_exception(
                httpx.TimeoutException("Request timed out"),
                url="https://api.rodiumai.io/v1/chat/completions",
                method="POST",
            )
        with pytest.raises(TimeoutError_):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
                timeout=0.001,
            )

    @pytest.mark.asyncio
    async def test_backend_error_message_included(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=400,
            json={"error": {"message": "Cannot read image.png (this model does not support image input)"}},
        )
        with pytest.raises(RodiumAIError) as exc_info:
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert "image.png" in str(exc_info.value)
        assert "Cannot read" in str(exc_info.value)
