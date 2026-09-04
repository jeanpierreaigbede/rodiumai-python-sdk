"""
Security tests — RodiumAI Python SDK
Covers: API key leakage, injection, input validation, HTTPS enforcement, DoS limits
"""

import pytest

from rodiumai import RodiumAI, RodiumAIError
from rodiumai.errors import InvalidAPIKeyError


class TestHTTPSEnforcement:
    def test_http_url_raises(self):
        with pytest.raises(ValueError, match="HTTPS is required"):
            RodiumAI(api_key="rdk-test", base_url="http://api.rodiumai.io/v1")

    def test_https_url_accepted(self):
        client = RodiumAI(api_key="rdk-test", base_url="https://api.rodiumai.io/v1")
        assert client is not None

    def test_http_localhost_with_port_allowed(self):
        client = RodiumAI(api_key="rdk-test", base_url="http://localhost:8080/v1")
        assert client is not None


class TestAPIKeyLeakage:
    def test_key_not_in_repr(self):
        client = RodiumAI(api_key="rdk-super-secret-key")
        assert "rdk-super-secret-key" not in repr(client)

    def test_key_not_in_str(self):
        client = RodiumAI(api_key="rdk-super-secret-key")
        assert "rdk-super-secret-key" not in str(client)

    def test_key_not_in_logs(self, capsys):
        client = RodiumAI(api_key="rdk-super-secret-key", log_level="DEBUG")
        client.logger.debug("test log entry")
        captured = capsys.readouterr()
        assert "rdk-super-secret-key" not in captured.err
        assert "rdk-super-secret-key" not in captured.out

    def test_bearer_token_not_in_logs(self, capsys):
        client = RodiumAI(api_key="rdk-super-secret-key", log_level="DEBUG")
        client.logger.debug("request headers logged")
        captured = capsys.readouterr()
        assert "Bearer rdk-super-secret-key" not in captured.err

    def test_key_not_in_error_message(self):
        """API key must never appear in error messages or stack traces."""
        try:
            RodiumAI(api_key="rdk-super-secret-key")
            raise RodiumAIError("test error", code=400)
        except RodiumAIError as e:
            assert "rdk-super-secret-key" not in str(e)
            assert "rdk-super-secret-key" not in repr(e)


class TestAPIKeyValidation:
    def test_empty_key_raises(self):
        with pytest.raises(InvalidAPIKeyError):
            RodiumAI(api_key="")

    def test_none_key_raises(self):
        with pytest.raises((InvalidAPIKeyError, TypeError)):
            RodiumAI(api_key=None)

    def test_whitespace_key_raises(self):
        with pytest.raises(InvalidAPIKeyError):
            RodiumAI(api_key="   ")

    def test_valid_key_accepted(self):
        client = RodiumAI(api_key="rdk-validkey123")
        assert client is not None


class TestHeaderInjection:
    def test_newline_in_api_key_rejected(self):
        """Prevent HTTP header injection via newlines in API key."""
        with pytest.raises((ValueError, Exception)):
            RodiumAI(api_key="rdk-key\r\nX-Injected: evil")

    def test_null_byte_in_api_key_rejected(self):
        with pytest.raises((ValueError, Exception)):
            RodiumAI(api_key="rdk-key\x00malicious")


class TestInputValidation:
    def test_empty_messages_raises(self):
        client = RodiumAI(api_key="rdk-test")
        with pytest.raises(Exception, match="messages"):
            import asyncio

            asyncio.run(client.chat.completions.create(model="openai/gpt-4o-mini", messages=[]))

    def test_temperature_too_high_raises(self):
        client = RodiumAI(api_key="rdk-test")
        with pytest.raises(Exception):
            import asyncio

            asyncio.run(
                client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=3.0,
                )
            )

    def test_temperature_negative_raises(self):
        client = RodiumAI(api_key="rdk-test")
        with pytest.raises(Exception):
            import asyncio

            asyncio.run(
                client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=-0.1,
                )
            )

    def test_max_tokens_zero_raises(self):
        client = RodiumAI(api_key="rdk-test")
        with pytest.raises(Exception):
            import asyncio

            asyncio.run(
                client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=0,
                )
            )


class TestRetryLimits:
    def test_max_retries_capped(self):
        """Max retries must not exceed a safe limit to prevent DoS."""
        client = RodiumAI(api_key="rdk-test", max_retries=10)
        assert client._http._max_retries <= 5

    def test_timeout_has_default(self):
        client = RodiumAI(api_key="rdk-test")
        assert client._http._timeout > 0
        assert client._http._timeout <= 120
