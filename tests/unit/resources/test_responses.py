import httpx
import pytest


class TestResponsesCreate:
    @pytest.mark.asyncio
    async def test_requires_model(self, client):
        with pytest.raises(ValueError, match="model is required"):
            await client.responses.create(model="", input="hi")

    @pytest.mark.asyncio
    async def test_parses_response_and_aggregates_output_text(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/responses",
            method="POST",
            json={
                "id": "resp_1",
                "object": "response",
                "model": "openai/gpt-4o",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Hel"},
                            {"type": "output_text", "text": "lo"},
                        ],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            },
        )

        res = await client.responses.create(model="openai/gpt-4o", input="hi")

        assert res.id == "resp_1"
        assert res.output_text == "Hello"
        assert res.usage.total_tokens == 6

    @pytest.mark.asyncio
    async def test_keeps_server_output_text_without_usage(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/responses",
            method="POST",
            json={
                "id": "resp_2",
                "output_text": "From server",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ignored"}],
                    }
                ],
                "rodiumai": {"cost_rodi": 0.5},
            },
        )

        res = await client.responses.create(model="openai/gpt-4o", input="hi")

        assert res.output_text == "From server"
        assert res.usage is None
        assert res.cost_rodi == 0.5
        assert res.model == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_aggregates_only_output_text_blocks(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/responses",
            method="POST",
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "refusal", "text": "nope"},
                            {"type": "output_text", "text": ""},
                            {"type": "output_text", "text": "yes"},
                        ],
                    }
                ],
            },
        )

        res = await client.responses.create(model="openai/gpt-4o", input="hi")

        assert res.output_text == "yes"
        assert res.output[0].content[0].type == "refusal"

    @pytest.mark.asyncio
    async def test_raises_mapped_error(self, httpx_mock, client):
        from rodiumai import ModelNotFoundError

        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/responses",
            method="POST",
            status_code=404,
            json={"error": {"message": "unknown model", "code": "model_not_found"}},
        )

        with pytest.raises(ModelNotFoundError, match="unknown model"):
            await client.responses.create(model="nope/model", input="hi")

    @pytest.mark.asyncio
    async def test_namespace_is_callable(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/responses",
            method="POST",
            json={"id": "resp_3", "output_text": "ok"},
        )

        res = await client.responses(model="openai/gpt-4o", input="hi")

        assert res.id == "resp_3"
        assert res.output_text == "ok"
        assert httpx_mock.get_request().method == "POST"


class TestResponsesStream:
    @pytest.mark.asyncio
    async def test_streams_output_text_delta_events(self, client, monkeypatch):
        from rodiumai.resources.responses import ResponseStreamEvent

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
                return {"X-Request-ID": "req-resp"}

            async def aread(self):
                return b""

            async def aiter_lines(self):
                yield 'data: {"type": "response.created", "response": {"id": "resp_1"}}'
                yield ""
                yield 'data: {"type": "response.output_text.delta", "delta": "Hel"}'
                yield ""
                yield 'data: {"type": "response.output_text.delta", "delta": "lo"}'
                yield ""
                yield 'data: {"type": "response.completed"}'

        class FakeHttpClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, method, url, **kwargs):
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeHttpClient())

        stream = await client.responses.create(
            model="openai/gpt-4o", input="hi", stream=True
        )

        events = []
        async for event in stream:
            assert isinstance(event, ResponseStreamEvent)
            events.append(event)

        text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        assert text == "Hello"
        assert events[-1].type == "response.completed"
