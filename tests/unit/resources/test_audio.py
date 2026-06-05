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
    async def test_speech_synthesis_returns_bytes(self, httpx_mock, client, mock_speech_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/speech",
            method="POST",
            json=mock_speech_response,
        )
        speech = await client.audio.speech.create(
            model="auto",
            input="Hello",
            voice="alloy",
        )
        assert isinstance(speech, SpeechResponse)
        assert isinstance(speech.content, bytes)
        assert speech.content_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_video_stub_raises_not_implemented(self):
        gen = Generations()
        with pytest.raises(NotImplementedError):
            await gen.create(model="auto", prompt="test")
