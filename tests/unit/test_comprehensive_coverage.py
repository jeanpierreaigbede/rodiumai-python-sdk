import pytest

from rodiumai import RodiumAI
from rodiumai.errors import (
    InsufficientRODIError,
    InternalServerError,
    NetworkError,
    RodiumAIError,
)
from rodiumai.logger import RodiumAILogger
from rodiumai.resources.audio import SpeechResponse, Transcription
from rodiumai.resources.images import ImagesResponse


class TestClientInitEdgeCases:
    def test_invalid_key_format_does_not_raise(self):
        client = RodiumAI(api_key="invalid-key")
        assert client._api_key == "invalid-key"

    def test_str_repr_masked(self):
        client = RodiumAI(api_key="rdk-somekey12345678")
        s = str(client)
        assert "****" in s
        assert "api_key" in s
        assert "base_url" in s

    def test_repr_masked(self):
        client = RodiumAI(api_key="rdk-anotherkey7890")
        r = repr(client)
        assert "****" in r
        assert "RodiumAI" in r

    def test_http_url_raises_error(self):
        with pytest.raises(ValueError, match="HTTPS"):
            RodiumAI(api_key="rdk-test", base_url="http://api.rodiumai.io/v1")


class TestUsageEdgeCases:
    def test_average_latency_no_requests(self):
        from rodiumai.usage import UsageStats

        us = UsageStats()
        assert us.average_latency_ms == 0.0

    def test_error_rate_no_requests(self):
        from rodiumai.usage import UsageStats

        us = UsageStats()
        assert us.error_rate == 0.0

    def test_recent_error_rate_empty(self):
        from rodiumai.usage import UsageStats

        us = UsageStats()
        assert us.recent_error_rate == 0.0

    def test_consecutive_errors_accumulates(self):
        from rodiumai.usage import UsageStats

        us = UsageStats()
        us.record_request(success=True, model="auto", endpoint="/chat", latency_ms=10)
        us.record_request(success=False, model="auto", endpoint="/chat", latency_ms=10)
        us.record_request(success=False, model="auto", endpoint="/chat", latency_ms=10)
        assert us.consecutive_errors == 2

    def test_to_dict_includes_all_fields(self):
        from rodiumai.usage import UsageStats

        us = UsageStats()
        us.record_request(
            success=True,
            model="auto",
            endpoint="/chat",
            latency_ms=10,
            prompt_tokens=5,
            completion_tokens=10,
        )
        d = us.to_dict()
        assert "total_requests" in d
        assert "successful_requests" in d
        assert "failed_requests" in d
        assert "total_tokens" in d
        assert d["total_tokens"] == 15
        assert d["average_latency_ms"] == 10.0
        assert d["error_rate"] == 0.0

    def test_reset_works(self):
        from rodiumai.usage import UsageStats

        us = UsageStats()
        us.record_request(success=True, model="auto", endpoint="/chat", latency_ms=10)
        us.reset()
        assert us.total_requests == 0
        assert us.average_latency_ms == 0.0

    def test_last_errors_overflow(self):
        from rodiumai.usage import UsageStats

        us = UsageStats()
        for _ in range(12):
            us.record_request(success=True, model="auto", endpoint="/chat", latency_ms=10)
        assert us.recent_error_rate == 0.0
        assert len(us._last_errors) == 10


class TestEmbeddingsEdgeCases:
    @pytest.mark.asyncio
    async def test_create_with_error(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/embeddings",
            method="POST",
            status_code=402,
        )
        with pytest.raises(InsufficientRODIError):
            await client.embeddings.create(
                model="auto",
                input="test text",
            )


class TestImagesEdgeCases:
    @pytest.mark.asyncio
    async def test_generate_with_quality(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/images/generations",
            method="POST",
            json={
                "created": 12345,
                "data": [
                    {"url": "https://example.com/img.png", "b64_json": None},
                ],
            },
        )
        response = await client.images.generate(
            model="pro",
            prompt="test image",
            n=1,
            size="1024x1024",
            quality="hd",
        )
        assert isinstance(response, ImagesResponse)
        assert response.data[0].url == "https://example.com/img.png"
        assert response.data[0].b64_json is None

    @pytest.mark.asyncio
    async def test_generate_with_b64_json(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/images/generations",
            method="POST",
            json={
                "created": 12345,
                "data": [
                    {"b64_json": "base64data", "url": None},
                ],
            },
        )
        response = await client.images.generate(
            model="auto",
            prompt="test",
        )
        assert response.data[0].b64_json == "base64data"
        assert response.data[0].url is None

    @pytest.mark.asyncio
    async def test_generate_with_error(self, httpx_mock, client):
        for _ in range(4):
            httpx_mock.add_response(
                url="https://api.rodiumai.io/v1/images/generations",
                method="POST",
                status_code=500,
            )
        with pytest.raises(InternalServerError):
            await client.images.generate(
                model="auto",
                prompt="test",
            )


class TestAudioEdgeCases:
    @pytest.mark.asyncio
    async def test_transcription_with_language(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/transcriptions",
            method="POST",
            json={"text": "hello world"},
        )
        result = await client.audio.transcriptions.create(
            model="auto",
            file=b"fake audio bytes",
            language="en",
        )
        assert isinstance(result, Transcription)
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_transcription_with_error(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/transcriptions",
            method="POST",
            status_code=402,
        )
        with pytest.raises(InsufficientRODIError):
            await client.audio.transcriptions.create(
                model="auto",
                file=b"fake",
            )

    @pytest.mark.asyncio
    async def test_speech_with_all_options(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/speech",
            method="POST",
            content=b"audio data",
            headers={"content-type": "audio/wav"},
        )
        result = await client.audio.speech.create(
            model="auto",
            input="Hello world",
            voice="nova",
            response_format="wav",
            speed=1.2,
        )
        assert isinstance(result, SpeechResponse)
        assert result.content_type == "audio/wav"

    @pytest.mark.asyncio
    async def test_speech_with_string_content(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/speech",
            method="POST",
            content=b"text data",
            headers={"content-type": "audio/mpeg"},
        )
        result = await client.audio.speech.create(
            model="auto",
            input="Hi",
            voice="alloy",
        )
        assert isinstance(result.content, bytes)
        assert result.content_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_speech_with_error(self, httpx_mock, client):
        for _ in range(4):
            httpx_mock.add_response(
                url="https://api.rodiumai.io/v1/audio/speech",
                method="POST",
                status_code=503,
            )
        from rodiumai.errors import ServiceUnavailableError

        with pytest.raises(ServiceUnavailableError):
            await client.audio.speech.create(
                model="auto",
                input="Hi",
                voice="alloy",
            )

    @pytest.mark.asyncio
    async def test_speech_with_missing_content(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/speech",
            method="POST",
            content=b"",
            headers={"content-type": "audio/mpeg"},
        )
        result = await client.audio.speech.create(
            model="auto",
            input="Hi",
            voice="alloy",
        )
        assert result.content == b""
        assert result.content_type == "audio/mpeg"


class TestChatStreamOptions:
    @pytest.mark.asyncio
    async def test_stream_with_max_tokens_top_p_stop(self, monkeypatch, client):
        chunks = [
            {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "created": 100,
                "model": "auto",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            },
        ]

        async def mock_stream(*args, **kwargs):
            for c in chunks:
                yield c

        monkeypatch.setattr(client._http, "_stream", mock_stream)
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=[".", "\n"],
            stream=True,
        )
        async for chunk in stream:
            assert chunk.choices[0].delta.content == "Hello"

    @pytest.mark.asyncio
    async def test_stream_final_chunk(self, monkeypatch, client):
        async def mock_stream(*args, **kwargs):
            yield {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "created": 100,
                "model": "auto",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        monkeypatch.setattr(client._http, "_stream", mock_stream)
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        async for chunk in stream:
            assert chunk.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_empty_messages_validation(self, client):
        with pytest.raises(ValueError, match="messages must not be empty"):
            await client.chat.completions.create(
                model="auto",
                messages=[],
            )


class TestHTTPEdgeCases:
    @pytest.mark.asyncio
    async def test_final_fallback_error(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=400,
        )
        with pytest.raises(RodiumAIError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @pytest.mark.asyncio
    async def test_network_error(self, httpx_mock, client):
        import httpx

        for _ in range(4):
            httpx_mock.add_exception(
                httpx.NetworkError("Connection refused"),
                url="https://api.rodiumai.io/v1/chat/completions",
                method="POST",
            )
        with pytest.raises(NetworkError):
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hi"}],
            )

    def test_mask_key_short(self):
        from rodiumai._http import AsyncHTTPClient

        assert AsyncHTTPClient._mask_key("abc") == "****"

    def test_mask_key_normal(self):
        from rodiumai._http import AsyncHTTPClient

        key = AsyncHTTPClient._mask_key("rdk-abcdefgh-12345678")
        assert key == "rdk-****5678"

    def test_get_usage_from_response_full(self):
        from rodiumai._http import AsyncHTTPClient

        client = AsyncHTTPClient(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        usage = client._get_usage_from_response(
            {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        )
        assert usage == {"prompt": 10, "completion": 20, "total": 30}

    def test_get_usage_from_response_none(self):
        from rodiumai._http import AsyncHTTPClient

        client = AsyncHTTPClient(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        assert client._get_usage_from_response({}) is None

    def test_get_usage_from_response_string(self):
        from rodiumai._http import AsyncHTTPClient

        client = AsyncHTTPClient(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        assert client._get_usage_from_response({"usage": "nope"}) is None


class TestBackendErrorMessage:
    @pytest.mark.asyncio
    async def test_backend_error_as_string(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=400,
            json={"error": "Model not found"},
        )
        with pytest.raises(RodiumAIError) as exc_info:
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert "Model not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_backend_error_on_401(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=401,
            json={"error": {"message": "Invalid API key provided"}},
        )
        with pytest.raises(RodiumAIError) as exc_info:
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert "Invalid API key" in str(exc_info.value)
        assert exc_info.value.code == 401


class TestClientKeyFormat:
    def test_invalid_format_triggers_warning(self, monkeypatch):
        from rodiumai import RodiumAI

        messages = []
        logger = RodiumAILogger("rodiumai", log_level="WARNING")

        def capture_alert(alert_type, message, **props):
            messages.append((alert_type, message))

        logger.log_alert = capture_alert
        with monkeypatch.context() as m:
            m.setattr("rodiumai.client.RodiumAILogger", lambda **kw: logger)
            RodiumAI(api_key="bad key with spaces")
        assert len(messages) > 0
        assert messages[0][0] == "invalid_api_key_format"

    def test_valid_key_no_warning(self):
        from rodiumai import RodiumAI

        client = RodiumAI(api_key="rdk-valid-key-12345")
        assert client._api_key == "rdk-valid-key-12345"


class TestNonDictBackendResponse:
    @pytest.mark.asyncio
    async def test_backend_error_response_as_array(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            status_code=400,
            json=["error", "details"],
        )
        with pytest.raises(RodiumAIError) as exc_info:
            await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert exc_info.value.error_code == "http_400"


class TestStreamEdgeCases:
    @pytest.mark.asyncio
    async def test_stream_empty_choices_get_skipped(self, monkeypatch, client):
        collected = []

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
                return {"X-Request-ID": "req-1"}

            async def aread(self):
                return b""

            async def aiter_lines(self):
                yield 'data: {"choices":[]}'
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}'
                yield "data: [DONE]"

        class FakeHttpClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, method, url, **kwargs):
                return FakeResponse()

        monkeypatch.setattr("httpx.AsyncClient", lambda **kw: FakeHttpClient())
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        async for chunk in stream:
            collected.append(chunk)
        assert len(collected) == 1
        assert collected[0].choices[0].delta.content == "ok"
