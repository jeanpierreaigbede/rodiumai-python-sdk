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
