class TestAudio:
    def test_transcription_returns_text(self):
        pass

    def test_speech_synthesis_returns_bytes(self):
        pass

    def test_supported_formats(self):
        pass

    def test_video_stub_raises_not_implemented(self):
        from rodiumai.resources.video import Generations
        import pytest
        gen = Generations()
        with pytest.raises(NotImplementedError):
            gen.create(model="auto", prompt="test")
