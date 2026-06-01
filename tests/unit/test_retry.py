import pytest

from rodiumai.errors import InternalServerError, InvalidAPIKeyError, ServiceUnavailableError


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_no_retry_on_401(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=401,
        )
        with pytest.raises(InvalidAPIKeyError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_402(self, httpx_mock, client):
        from rodiumai.errors import InsufficientRODIError
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=402,
        )
        with pytest.raises(InsufficientRODIError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_403(self, httpx_mock, client):
        from rodiumai.errors import PermissionDeniedError
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=403,
        )
        with pytest.raises(PermissionDeniedError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_404(self, httpx_mock, client):
        from rodiumai.errors import ModelNotFoundError
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=404,
        )
        with pytest.raises(ModelNotFoundError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_retry_on_429_then_succeeds(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=429,
            headers={"Retry-After": "0.01"},
        )
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert response.id == "chatcmpl-123"
        assert len(httpx_mock.get_requests()) == 2

    @pytest.mark.asyncio
    async def test_retry_on_500_then_succeeds(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=500,
        )
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert response.id == "chatcmpl-123"
        assert len(httpx_mock.get_requests()) == 2

    @pytest.mark.asyncio
    async def test_retry_3_times_on_500_then_raises(self, httpx_mock, client):
        for _ in range(4):
            httpx_mock.add_response(
                url="https://api.rodiumai.io/v1/chat/completions",
                method="POST",
                status_code=500,
            )
        with pytest.raises(InternalServerError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hello"}],
            )
        requests = httpx_mock.get_requests()
        assert len(requests) == 4
