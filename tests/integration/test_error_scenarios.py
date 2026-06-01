class TestErrorScenarios:
    def test_server_401_raises_invalid_api_key(self):
        pass

    def test_server_402_raises_insufficient_rodi(self):
        pass

    def test_server_429_with_retry_after_raises_rate_limit(self):
        pass

    def test_server_500_retried_3_times_then_internal_server_error(self):
        pass

    def test_server_503_retried_3_times_then_service_unavailable(self):
        pass

    def test_network_drop_mid_streaming_raises_network_error(self):
        pass

    def test_server_slow_raises_timeout_error(self):
        pass
