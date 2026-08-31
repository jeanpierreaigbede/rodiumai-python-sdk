import pytest

from rodiumai.resources.audio import SpeechResponse, Transcription
from rodiumai.resources.video import Generations


class TestAudio:
    @pytest.mark.asyncio
    async def test_transcription_returns_text(
        self, httpx_mock, client, mock_transcription_response
    ):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/transcriptions",
            method="POST",
            json=mock_transcription_response,
        )
        transcript = await client.audio.transcriptions.create(
            model="auto",
            file=b"fake-audio",
        )
        assert isinstance(transcript, Transcription)
        assert transcript.text == "Hello, world."

    @pytest.mark.asyncio
    async def test_speech_synthesis_returns_bytes(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/speech",
            method="POST",
            content=b"fake-audio-bytes",
            headers={"content-type": "audio/mpeg"},
        )
        speech = await client.audio.speech.create(
            model="openai/gpt-4o",
            input="Hello",
            voice="alloy",
        )
        assert isinstance(speech, SpeechResponse)
        assert isinstance(speech.content, bytes)
        assert speech.content == b"fake-audio-bytes"
        assert speech.content_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_video_generations(self, httpx_mock, client):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/videos/generations",
            method="POST",
            json={"created": 1, "data": [{"url": "https://example.com/v.mp4"}]},
        )
        from rodiumai.resources.video import Generations

        gen = Generations(client._http)
        result = await gen.create(model="openai/gpt-4o", prompt="test")
        assert result.data[0].url == "https://example.com/v.mp4"
