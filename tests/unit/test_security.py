from rodiumai import RodiumAI


class TestSecurity:
    def test_api_key_masked_in_repr(self):
        client = RodiumAI(api_key="rdk-my-secret-key")
        assert "rdk-my-secret-key" not in repr(client)
        assert "rdk-" in repr(client)

    def test_api_key_masked_in_str(self):
        client = RodiumAI(api_key="rdk-my-secret-key")
        assert "rdk-my-secret-key" not in str(client)

    def test_api_key_never_in_logs(self, capsys):
        client = RodiumAI(api_key="rdk-my-secret-key", log_level="INFO")
        client.logger.info("test log")
        captured = capsys.readouterr()
        assert "rdk-my-secret-key" not in captured.err

    def test_http_rejected_https_accepted(self):
        with RodiumAI(api_key="rdk-test", base_url="https://api.rodiumai.io/v1"):
            pass

    def test_http_url_raises_error(self):
        import pytest
        with pytest.raises(ValueError, match="HTTPS is required"):
            RodiumAI(api_key="rdk-test", base_url="http://api.rodiumai.io/v1")

    def test_authorization_header_not_logged(self, capsys):
        client = RodiumAI(api_key="rdk-should-not-appear", log_level="INFO")
        client.logger.info("request made")
        captured = capsys.readouterr()
        assert "Bearer rdk-" not in captured.err
        assert "rdk-should-not-appear" not in captured.err
