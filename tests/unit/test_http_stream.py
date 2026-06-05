import httpx
import pytest

from rodiumai.errors import InternalServerError, RateLimitError, TimeoutError_


class TestHTTPRetryParsing:
    @pytest.mark.asyncio
    async def test_429_with_invalid_retry_after(self, httpx_mock, client):
        for _ in range(4):
            httpx_mock.add_response(
                url="https://api.rodiumai.io/v1/chat/completions",
                method="POST",
                status_code=429,
                headers={"Retry-After": "not-a-number"},
            )
        with pytest.raises(RateLimitError) as exc_info:
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert exc_info.value.retry_after == 1.0

    @pytest.mark.asyncio
    async def test_429_missing_retry_after_defaults(self, httpx_mock, client):
        for _ in range(4):
            httpx_mock.add_response(
                url="https://api.rodiumai.io/v1/chat/completions",
                method="POST",
                status_code=429,
            )
        with pytest.raises(RateLimitError) as exc_info:
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert exc_info.value.retry_after == 1.0


class TestStreamDirect:
    @pytest.mark.asyncio
    async def test_stream_parses_sse(self, monkeypatch):
        from rodiumai._http import AsyncHTTPClient

        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            @property
            def status_code(self):
                return 200

            @property
            def headers(self):
                return {"X-Request-ID": "req-stream"}

            async def aread(self):
                return b""

            async def aiter_lines(self):
                yield 'data: {"content": "hello"}'
                yield "data: [DONE]"
                yield 'data: {"content": "should not appear"}'

        class FakeHttpClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, method, url, **kwargs):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeHttpClient())
        client = AsyncHTTPClient(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        collected = []
        async for chunk in client._stream("POST", "/chat/completions", json_body={}):
            collected.append(chunk)
        assert len(collected) == 1
        assert collected[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_stream_raises_on_error_status(self, monkeypatch):
        from rodiumai._http import AsyncHTTPClient

        class FakeErrorResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            @property
            def status_code(self):
                return 500

            @property
            def headers(self):
                return {"X-Request-ID": "req-err"}

            async def aread(self):
                return b"{}"

            async def aiter_lines(self):
                yield ""
                return

        class FakeHttpClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, method, url, **kwargs):
                return FakeErrorResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeHttpClient())
        client = AsyncHTTPClient(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        with pytest.raises(InternalServerError):
            async for _ in client._stream("POST", "/chat/completions", json_body={}):
                pass

    @pytest.mark.asyncio
    async def test_stream_non_data_lines_skipped(self, monkeypatch):
        from rodiumai._http import AsyncHTTPClient

        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            @property
            def status_code(self):
                return 200

            @property
            def headers(self):
                return {"X-Request-ID": "req-skip"}

            async def aread(self):
                return b""

            async def aiter_lines(self):
                yield ""
                yield "event: message"
                yield 'data: {"content": "real"}'
                yield ""

        class FakeHttpClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, method, url, **kwargs):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeHttpClient())
        client = AsyncHTTPClient(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        collected = []
        async for chunk in client._stream("POST", "/chat/completions", json_body={}):
            collected.append(chunk)
        assert len(collected) == 1
        assert collected[0]["content"] == "real"

    @pytest.mark.asyncio
    async def test_stream_empty_payload_skipped(self, monkeypatch):
        from rodiumai._http import AsyncHTTPClient

        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            @property
            def status_code(self):
                return 200

            @property
            def headers(self):
                return {"X-Request-ID": "req-empty"}

            async def aread(self):
                return b""

            async def aiter_lines(self):
                yield "data: "
                yield "data:  "
                yield 'data: {"content": "ok"}'
                yield "data: [DONE]"

        class FakeHttpClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, method, url, **kwargs):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeHttpClient())
        client = AsyncHTTPClient(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        collected = []
        async for chunk in client._stream("POST", "/chat/completions", json_body={}):
            collected.append(chunk)
        assert len(collected) == 1
        assert collected[0]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_timeout_retry_then_raises(self, client, monkeypatch):
        call_count = 0

        class FakeTimeoutContext:
            async def __aenter__(self):
                nonlocal call_count
                call_count += 1
                raise httpx.TimeoutException("timed out")

            async def __aexit__(self, *args):
                pass

        class FakeHttpClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, method, url, **kwargs):
                return FakeTimeoutContext()

        monkeypatch.setattr(client._http._client, "stream", FakeHttpClient().stream)
        with pytest.raises(TimeoutError_):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert call_count == 4
