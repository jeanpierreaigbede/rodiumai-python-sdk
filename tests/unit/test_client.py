import pytest

from rodiumai import InvalidAPIKeyError, RodiumAI
from rodiumai._version import SDK_IDENTIFIER, VERSION


class TestClientInit:
    def test_valid_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("RODIUMAI_API_KEY", "rdk-test-key")
        client = RodiumAI()
        assert client._api_key == "rdk-test-key"

    def test_valid_api_key_passed_directly(self):
        client = RodiumAI(api_key="rdk-test-key")
        assert client._api_key == "rdk-test-key"

    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("RODIUMAI_API_KEY", raising=False)
        with pytest.raises(InvalidAPIKeyError):
            RodiumAI()

    def test_http_url_raises_value_error(self):
        with pytest.raises(ValueError, match="HTTPS is required"):
            RodiumAI(api_key="rdk-test-key", base_url="http://api.rodiumai.io/v1")

    def test_custom_timeout(self):
        client = RodiumAI(api_key="rdk-test-key", timeout=60.0)
        assert client._timeout == 60.0

    def test_custom_base_url(self):
        client = RodiumAI(api_key="rdk-test-key", base_url="https://custom.rodiumai.io/v1")
        assert client._base_url == "https://custom.rodiumai.io/v1"

    def test_repr_masks_api_key(self):
        client = RodiumAI(api_key="rdk-secret-key-99999")
        assert "rdk-secret-key-99999" not in repr(client)
        assert "****" in repr(client)

    def test_str_masks_api_key(self):
        client = RodiumAI(api_key="rdk-secret-key-99999")
        assert "rdk-secret-key-99999" not in str(client)
        assert "****" in str(client)

    def test_sdk_headers(self):
        client = RodiumAI(api_key="rdk-test-key")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer rdk-test-key"
        assert headers["X-RodiumAI-SDK"] == SDK_IDENTIFIER
        assert headers["X-RodiumAI-Version"] == VERSION

    def test_usage_property(self):
        client = RodiumAI(api_key="rdk-test-key")
        assert client.usage.total_requests == 0

    def test_resources_initialized(self):
        client = RodiumAI(api_key="rdk-test-key")
        assert client.chat is not None
        assert client.embeddings is not None
        assert client.images is not None
        assert client.audio is not None
        assert client.video is not None
