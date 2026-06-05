from rodiumai.usage import UsageStats


class TestUsageStats:
    def test_initial_state(self):
        stats = UsageStats()
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.total_tokens == 0

    def test_incremented_after_successful_request(self):
        stats = UsageStats()
        stats.record_request(
            success=True,
            model="auto",
            endpoint="/v1/chat/completions",
            latency_ms=100,
            prompt_tokens=10,
            completion_tokens=20,
        )
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.failed_requests == 0
        assert stats.total_tokens == 30

    def test_incremented_after_failed_request(self):
        stats = UsageStats()
        stats.record_request(
            success=False,
            model="auto",
            endpoint="/v1/chat/completions",
            latency_ms=50,
        )
        assert stats.total_requests == 1
        assert stats.successful_requests == 0
        assert stats.failed_requests == 1

    def test_error_rate_calculation(self):
        stats = UsageStats()
        stats.record_request(success=True, model="auto", endpoint="/test", latency_ms=10)
        stats.record_request(success=False, model="auto", endpoint="/test", latency_ms=10)
        assert stats.error_rate == 0.5

    def test_tokens_accumulated_correctly(self):
        stats = UsageStats()
        stats.record_request(
            success=True,
            model="auto",
            endpoint="/test",
            latency_ms=10,
            prompt_tokens=50,
            completion_tokens=50,
        )
        stats.record_request(
            success=True,
            model="auto",
            endpoint="/test",
            latency_ms=10,
            prompt_tokens=100,
            completion_tokens=100,
        )
        assert stats.total_tokens == 300
        assert stats.total_prompt_tokens == 150
        assert stats.total_completion_tokens == 150

    def test_reset_clears_stats(self):
        stats = UsageStats()
        stats.record_request(success=True, model="auto", endpoint="/test", latency_ms=10)
        stats.reset()
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0

    def test_average_latency(self):
        stats = UsageStats()
        stats.record_request(success=True, model="auto", endpoint="/test", latency_ms=100)
        stats.record_request(success=True, model="auto", endpoint="/test", latency_ms=200)
        assert stats.average_latency_ms == 150.0

    def test_to_dict_contains_all_fields(self):
        stats = UsageStats()
        stats.record_request(success=True, model="auto", endpoint="/test", latency_ms=100)
        d = stats.to_dict()
        assert "total_requests" in d
        assert "error_rate" in d
        assert "requests_by_model" in d
        assert d["requests_by_model"] == {"auto": 1}

    def test_recent_error_rate(self):
        stats = UsageStats()
        assert stats.recent_error_rate == 0.0
        stats.record_request(success=False, model="auto", endpoint="/test", latency_ms=10)
        assert stats.recent_error_rate == 1.0
