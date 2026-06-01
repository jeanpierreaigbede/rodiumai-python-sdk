import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from ._version import VERSION


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "props"):
            log_entry.update(record.props)
        return json.dumps(log_entry)


_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _resolve_log_level(log_level: Optional[str] = None) -> int:
    if log_level is not None:
        return _LOG_LEVELS.get(log_level.upper(), logging.WARNING)
    env_level = os.environ.get("RODIUMAI_LOG_LEVEL", "WARNING")
    return _LOG_LEVELS.get(env_level.upper(), logging.WARNING)


class RodiumAILogger:
    def __init__(self, name: str = "rodiumai", log_level: Optional[str] = None):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(_resolve_log_level(log_level))
        self._logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        self._logger.addHandler(handler)
        self._logger.propagate = False

    def _log(self, level: int, message: str, **props: Any):
        if self._logger.isEnabledFor(level):
            record = self._logger.makeRecord(
                self._logger.name,
                level,
                "", 0, message, (), None,
            )
            record.props = props
            self._logger.handle(record)

    def debug(self, message: str, **props: Any):
        self._log(logging.DEBUG, message, **props)

    def info(self, message: str, **props: Any):
        self._log(logging.INFO, message, **props)

    def warning(self, message: str, **props: Any):
        self._log(logging.WARNING, message, **props)

    def error(self, message: str, **props: Any):
        self._log(logging.ERROR, message, **props)

    def log_request(
        self,
        request_id: str,
        model: str,
        endpoint: str,
        method: str,
        latency_ms: float,
        status: str,
        http_status: int,
        tokens: Optional[dict] = None,
        retry_count: int = 0,
        streaming: bool = False,
        error_code: Optional[str] = None,
    ):
        self.info(
            f"{method} {endpoint} -> {http_status}",
            request_id=request_id,
            sdk_version=VERSION,
            model=model,
            endpoint=endpoint,
            method=method,
            latency_ms=round(latency_ms, 2),
            tokens=tokens or {"prompt": 0, "completion": 0, "total": 0},
            status=status,
            http_status=http_status,
            error_code=error_code,
            retry_count=retry_count,
            streaming=streaming,
        )

    def log_alert(self, alert_type: str, message: str, **props: Any):
        self.warning(f"ALERT [{alert_type}]: {message}", alert_type=alert_type, **props)
