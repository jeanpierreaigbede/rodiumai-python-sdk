from dataclasses import dataclass, field
from typing import Dict


@dataclass
class UsageStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    requests_by_model: Dict[str, int] = field(default_factory=dict)
    requests_by_endpoint: Dict[str, int] = field(default_factory=dict)
    _consecutive_errors: int = 0
    _last_errors: list[int] = field(default_factory=list)

    @property
    def average_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_requests, 2)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.failed_requests / self.total_requests, 3)

    def record_request(
        self,
        success: bool,
        model: str,
        endpoint: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self._consecutive_errors = 0
        else:
            self.failed_requests += 1
            self._consecutive_errors += 1

        self.total_tokens += prompt_tokens + completion_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_latency_ms += latency_ms

        self.requests_by_model[model] = self.requests_by_model.get(model, 0) + 1
        self.requests_by_endpoint[endpoint] = self.requests_by_endpoint.get(endpoint, 0) + 1

        self._last_errors.append(0 if success else 1)
        if len(self._last_errors) > 10:
            self._last_errors.pop(0)

    @property
    def recent_error_rate(self) -> float:
        if not self._last_errors:
            return 0.0
        return sum(self._last_errors) / len(self._last_errors)

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "average_latency_ms": self.average_latency_ms,
            "error_rate": self.error_rate,
            "requests_by_model": dict(self.requests_by_model),
            "requests_by_endpoint": dict(self.requests_by_endpoint),
        }

    def reset(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_ms = 0.0
        self.requests_by_model.clear()
        self.requests_by_endpoint.clear()
        self._consecutive_errors = 0
        self._last_errors.clear()
