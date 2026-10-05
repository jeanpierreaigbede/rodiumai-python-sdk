import httpx
import pytest


class TestMessagesCreate:
    @pytest.mark.asyncio
    async def test_sends_anthropic_headers_and_parses_response(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/messages",
            method="POST",
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "anthropic/claude",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        )

        res = await client.messages.create(
            model="anthropic/claude",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert res.id == "msg_1"
        assert res.text == "hello"
        assert res.content[0].text == "hello"
        assert res.usage.input_tokens == 5

        request = httpx_mock.get_request()
        assert request.headers.get("x-api-key") == "rdk-test-key-12345"
        assert request.headers.get("anthropic-version") == "2023-06-01"


class TestMessagesStream:
    @pytest.mark.asyncio
    async def test_streams_anthropic_events_without_done_sentinel(self, client, monkeypatch):
        from rodiumai.resources.messages import MessageStreamEvent

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
                return {"X-Request-ID": "req-msg"}

            async def aread(self):
                return b""

            async def aiter_lines(self):
                yield 'event: message_start'
                yield 'data: {"type": "message_start", "message": {"id": "msg_1"}}'
                yield ""
                yield 'event: content_block_delta'
                yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}'
                yield ""
                yield 'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}'
                yield ""
                yield 'data: {"type": "message_stop"}'

        class FakeHttpClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream(self, method, url, **kwargs):
                # The Anthropic headers must be forwarded on the stream request.
                assert kwargs["headers"].get("x-api-key") == "rdk-test-key-12345"
                assert kwargs["headers"].get("anthropic-version") == "2023-06-01"
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeHttpClient())

        stream = await client.messages.create(
            model="anthropic/claude",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )

        events = []
        async for event in stream:
            assert isinstance(event, MessageStreamEvent)
            events.append(event)

        assert [e.type for e in events] == [
            "message_start",
            "content_block_delta",
            "content_block_delta",
            "message_stop",
        ]
        text = "".join(
            e.delta["text"] for e in events if e.type == "content_block_delta"
        )
        assert text == "Hello"
