import pytest
import logging
import os

from rodiumai.logger import RodiumAILogger, _resolve_log_level, JSONFormatter


class TestLoggerEdgeCases:
    def test_debug_level(self):
        logger = RodiumAILogger("test-debug", log_level="DEBUG")
        logger.debug("debug test")
        # No assertion needed - just ensure it doesn't crash

    def test_error_level(self):
        logger = RodiumAILogger("test-error", log_level="ERROR")
        logger.error("error test")
        # No assertion needed - just ensure it doesn't crash

    def test_log_alert(self):
        logger = RodiumAILogger("test-alert", log_level="INFO")
        logger.log_alert("test_alert", "alert message")
        # No assertion needed - just ensure it doesn't crash

    def test_resolve_log_level_custom(self):
        assert _resolve_log_level("DEBUG") == logging.DEBUG

    def test_resolve_log_level_env(self, monkeypatch):
        monkeypatch.setenv("RODIUMAI_LOG_LEVEL", "ERROR")
        assert _resolve_log_level(None) == logging.ERROR

    def test_resolve_log_level_invalid(self):
        assert _resolve_log_level("INVALID") == logging.WARNING

    def test_json_formatter_with_props(self):
        import json
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg="test msg",
            args=(), exc_info=None,
        )
        record.props = {"custom": "value"}
        formatter = JSONFormatter()
        result = json.loads(formatter.format(record))
        assert result["custom"] == "value"
        assert result["level"] == "INFO"
        assert result["message"] == "test msg"
