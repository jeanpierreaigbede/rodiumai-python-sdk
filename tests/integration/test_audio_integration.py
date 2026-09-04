import pytest


class TestAudioIntegration:
    @pytest.mark.asyncio
    async def test_transcriptions_endpoint(self, httpx_mock, client, mock_transcription_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/transcriptions",
            method="POST",
            json=mock_transcription_response,
        )
        transcript = await client.audio.transcriptions.create(
            model="auto",
            file=b"fake-audio-data",
        )
        assert transcript.text == "Hello, world."

    @pytest.mark.asyncio
    async def test_speech_endpoint(self, httpx_mock, client, mock_speech_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/audio/speech",
            method="POST",
            content=mock_speech_response["content"].encode(),
            headers={"content-type": mock_speech_response["content_type"]},
        )
        speech = await client.audio.speech.create(
            model="auto",
            input="Hello",
            voice="alloy",
        )
        assert len(speech.content) > 0
        assert speech.content_type == "audio/mpeg"
