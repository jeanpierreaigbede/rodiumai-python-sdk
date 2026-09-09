import pytest

from rodiumai._http import _extract_error_info
from rodiumai.resources.chat import ChatCompletion


class TestFlatAPI:
    @pytest.mark.asyncio
    async def test_flat_chat_string(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        response = await client.chat("Hello")
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_fluent_builder_chat(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        response = await (
            client.model("openai/gpt-4o")
            .temperature(0.7)
            .top_p(0.9)
            .max_tokens(100)
            .system_prompt("You are helpful.")
            .chat("Hi")
        )
        assert response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_client_stream(self, monkeypatch, client):
        async def mock_create(**kwargs):
            assert kwargs["stream"] is True

            async def gen():
                from rodiumai.resources.chat import ChatCompletionChunk, ChunkChoice, Delta

                yield ChatCompletionChunk(
                    id="c1",
                    choices=[ChunkChoice(index=0, delta=Delta(content="Hi"))],
                )

            return gen()

        monkeypatch.setattr(client.chat.completions, "create", mock_create)
        parts = []
        async for delta in client.stream("Hello"):
            parts.append(delta)
        assert parts == ["Hi"]

    @pytest.mark.asyncio
    async def test_model_info(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/models/openai/gpt-4o",
            method="GET",
            json={"id": "openai/gpt-4o"},
        )
        info = await client.model_info("openai/gpt-4o")
        assert info["id"] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_coding_models(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/models/coding",
            method="GET",
            json={"data": []},
        )
        result = await client.coding_models()
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_videos_flat(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/videos/generations",
            method="POST",
            json={"created": 1, "data": [{"url": "https://example.com/v.mp4"}]},
        )
        result = await client.videos(prompt="waves", duration_seconds=8)
        assert result.data[0].url == "https://example.com/v.mp4"

    @pytest.mark.asyncio
    async def test_transcribe_flat(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/transcriptions",
            method="POST",
            json={"text": "bonjour"},
        )
        result = await client.transcribe(b"audio", language="fr")
        assert result.text == "bonjour"

    @pytest.mark.asyncio
    async def test_speech_flat_returns_bytes(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/speech",
            method="POST",
            content=b"mp3-bytes",
            headers={"content-type": "audio/mpeg"},
        )
        result = await client.speech(input="Hello", voice="alloy")
        assert result == b"mp3-bytes"

    @pytest.mark.asyncio
    async def test_wallet(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/wallet",
            method="GET",
            json={"balance_rodi": "100"},
        )
        wallet = await client.wallet()
        assert wallet["balance_rodi"] == "100"

    @pytest.mark.asyncio
    async def test_pricing_all_and_filtered(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/pricing",
            method="GET",
            json={"data": []},
        )
        all_pricing = await client.pricing()
        assert all_pricing["data"] == []

        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/pricing",
            method="GET",
            json={"data": [{"model": "openai/gpt-4o"}]},
        )
        one = await client.pricing(model="openai/gpt-4o")
        assert one["data"][0]["model"] == "openai/gpt-4o"


class TestModelsNamespace:
    @pytest.mark.asyncio
    async def test_models_callable_and_nested(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/models",
            method="GET",
            json={"data": [{"id": "openai/gpt-4o"}]},
        )
        listed = await client.models()
        assert listed["data"][0]["id"] == "openai/gpt-4o"

        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/models",
            method="GET",
            json={"data": []},
        )
        nested = await client.models.list()
        assert nested["data"] == []

        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/models/anthropic/claude",
            method="GET",
            json={"id": "anthropic/claude"},
        )
        detail = await client.models.retrieve("anthropic/claude")
        assert detail["id"] == "anthropic/claude"

        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/models/coding",
            method="GET",
            json={"data": [{"id": "coding-model"}]},
        )
        coding = await client.models.list_coding()
        assert coding["data"][0]["id"] == "coding-model"


class TestMessagesNamespace:
    @pytest.mark.asyncio
    async def test_messages_create_and_callable(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/messages",
            method="POST",
            json={"content": [{"type": "text", "text": "OK"}]},
        )
        payload = {
            "model": "anthropic/claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        created = await client.messages.create(**payload)
        assert created.content[0].text == "OK"
        assert created.text == "OK"

        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/messages",
            method="POST",
            json={"content": [{"type": "text", "text": "Callable"}]},
        )
        called = await client.messages(**payload)
        assert called.content[0].text == "Callable"


class TestEmbeddingsAndImagesCallable:
    @pytest.mark.asyncio
    async def test_embeddings_callable(self, httpx_mock, client, mock_embedding_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/embeddings",
            method="POST",
            json=mock_embedding_response,
        )
        result = await client.embeddings("hello")
        assert len(result.data[0].embedding) == 3

    @pytest.mark.asyncio
    async def test_images_callable(self, httpx_mock, client, mock_image_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/images/generations",
            method="POST",
            json=mock_image_response,
        )
        result = await client.images(prompt="sunset")
        assert result.data[0].url.endswith(".png")


class TestVideoNamespace:
    @pytest.mark.asyncio
    async def test_video_callable(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/videos/generations",
            method="POST",
            json={"created": 2, "data": [{"b64_json": "abc"}]},
        )
        result = await client.video(prompt="test clip")
        assert result.data[0].b64_json == "abc"


class TestAudioPrepareFile:
    @pytest.mark.asyncio
    async def test_transcribe_from_path(self, httpx_mock, client, tmp_path):
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"wav-data")
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/transcriptions",
            method="POST",
            json={"text": "from path"},
        )
        result = await client.audio.transcriptions.create(model="auto", file=str(audio_file))
        assert result.text == "from path"


class TestHTTPHelpers:
    def test_extract_error_info_anthropic(self):
        msg, code = _extract_error_info(
            {"type": "error", "error": {"message": "bad request", "type": "invalid_request"}}
        )
        assert msg == "bad request"
        assert code == "invalid_request"

    def test_mask_key_rd_sk_prefix(self):
        from rodiumai._http import AsyncHTTPClient

        masked = AsyncHTTPClient._mask_key("rd_sk_abcdefghijklmnop")
        assert masked.startswith("rd_sk_****")

    def test_get_usage_partial_tokens(self):
        from rodiumai._http import AsyncHTTPClient

        client = AsyncHTTPClient(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        usage = client._get_usage_from_response({"usage": {"total_tokens": 5}})
        assert usage == {"prompt": 0, "completion": 0, "total": 5}

    @pytest.mark.asyncio
    async def test_request_binary_error_with_backend_code(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/speech",
            method="POST",
            status_code=400,
            json={"error": {"message": "bad speech", "code": "invalid_input"}},
        )
        from rodiumai.errors import RodiumAIError

        with pytest.raises(RodiumAIError) as exc_info:
            await client.audio.speech.create(model="auto", input="Hi", voice="alloy")
        assert exc_info.value.message == "bad speech"
        assert exc_info.value.error_code == "invalid_input"

    @pytest.mark.asyncio
    async def test_json_request_without_files(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hi"}],
        )
        req = httpx_mock.get_request()
        assert req.headers.get("content-type", "").startswith("application/json")


class TestRemainingCoverage:
    @pytest.mark.asyncio
    async def test_chat_with_message_list_and_options(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        response = await client.chat(
            [{"role": "user", "content": "Hi"}],
            temperature=0.2,
            top_p=0.8,
            max_tokens=50,
            timeout=15.0,
        )
        assert response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_chat_cost_rodi_from_rodiumai_block(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json={
                "id": "x",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "rodiumai": {"cost_rodi": 0.42},
            },
        )
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.cost_rodi == 0.42

    @pytest.mark.asyncio
    async def test_resource_error_paths(self, httpx_mock, client):
        from rodiumai.errors import InsufficientRODIError

        for url, method, call in [
            (
                "https://api.rodiumai.io/v1/wallet",
                "GET",
                lambda: client.wallet(),
            ),
            (
                "https://api.rodiumai.io/v1/pricing",
                "GET",
                lambda: client.pricing(),
            ),
            (
                "https://api.rodiumai.io/v1/models",
                "GET",
                lambda: client.models.list(),
            ),
            (
                "https://api.rodiumai.io/v1/models/openai/gpt-4o",
                "GET",
                lambda: client.models.retrieve("openai/gpt-4o"),
            ),
            (
                "https://api.rodiumai.io/v1/models/coding",
                "GET",
                lambda: client.models.list_coding(),
            ),
            (
                "https://api.rodiumai.io/v1/messages",
                "POST",
                lambda: client.messages.create(
                    model="anthropic/claude",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "x"}],
                ),
            ),
            (
                "https://api.rodiumai.io/v1/videos/generations",
                "POST",
                lambda: client.video.generations.create(model="auto", prompt="x"),
            ),
        ]:
            httpx_mock.add_response(url=url, method=method, status_code=402)
            with pytest.raises(InsufficientRODIError):
                await call()

    def test_prepare_file_passthrough(self):
        from rodiumai.resources.audio import Transcriptions

        sentinel = object()
        assert Transcriptions._prepare_file(sentinel) is sentinel
    async def test_system_prompt_with_message_list(self, httpx_mock, client, mock_chat_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        response = await client.system_prompt("Be concise.").chat(
            [{"role": "user", "content": "Hi"}]
        )
        assert response.choices[0].message.content

    def test_mask_key_generic_prefix(self):
        from rodiumai._http import AsyncHTTPClient

        assert AsyncHTTPClient._mask_key("abcd1234567890") == "abcd****7890"

    def test_get_usage_all_none_fields(self):
        from rodiumai._http import AsyncHTTPClient

        http = AsyncHTTPClient(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        assert http._get_usage_from_response({"usage": {}}) is None

    @pytest.mark.asyncio
    async def test_legacy_chat_wrapper_callable(self, httpx_mock, client, mock_chat_response):
        from rodiumai.resources.chat import Chat

        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/chat/completions",
            method="POST",
            json=mock_chat_response,
        )
        chat = Chat(client._http, client)
        response = await chat("Hello via legacy Chat")
        assert response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_stream_delta_with_role_only(self, monkeypatch, client):
        async def mock_stream(*args, **kwargs):
            yield {
                "id": "c1",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }

        monkeypatch.setattr(client._http, "_stream", mock_stream)
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert chunks[0].choices[0].delta.role == "assistant"
