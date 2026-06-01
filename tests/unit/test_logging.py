import json

from rodiumai.logger import RodiumAILogger


class TestLogging:
    def test_json_structure_validated(self, capsys):
        logger = RodiumAILogger(log_level="INFO")
        logger.info("test message", request_id="req-1")
        captured = capsys.readouterr()
        log_data = json.loads(captured.err.strip())
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "test message"
        assert log_data["request_id"] == "req-1"

    def test_log_level_filtering(self, capsys):
        logger = RodiumAILogger(log_level="ERROR")
        logger.info("should not appear")
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_retry_attempt_logged_at_warning(self, capsys):
        logger = RodiumAILogger(log_level="WARNING")
        logger.warning("Retry attempt 1/3", retry_count=1)
        captured = capsys.readouterr()
        assert "Retry attempt" in captured.err
        log_data = json.loads(captured.err.strip())
        assert log_data["level"] == "WARNING"

    def test_log_request_contains_required_fields(self, capsys):
        logger = RodiumAILogger(log_level="INFO")
        logger.log_request(
            request_id="req-1",
            model="auto",
            endpoint="/v1/chat/completions",
            method="POST",
            latency_ms=342,
            status="success",
            http_status=200,
            tokens={"prompt": 120, "completion": 85, "total": 205},
        )
        captured = capsys.readouterr()
        log_data = json.loads(captured.err.strip())
        assert log_data["request_id"] == "req-1"
        assert log_data["model"] == "auto"
        assert log_data["endpoint"] == "/v1/chat/completions"
        assert log_data["http_status"] == 200
