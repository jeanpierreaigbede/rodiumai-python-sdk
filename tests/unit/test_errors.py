import pytest

from rodiumai.errors import (
    InsufficientRODIError,
    InternalServerError,
    InvalidAPIKeyError,
    ModelNotFoundError,
    NetworkError,
    PermissionDeniedError,
    RateLimitError,
    RodiumAIError,
    ServiceUnavailableError,
    TimeoutError_,
    map_http_status,
)


class TestErrorHierarchy:
    def test_base_error(self):
        err = RodiumAIError("test", code=400, error_code="custom_error", request_id="req-123")
        assert err.message == "test"
        assert err.code == 400
        assert err.error_code == "custom_error"
        assert err.request_id == "req-123"
        assert err.fix_suggestion is None

    def test_invalid_api_key(self):
        err = InvalidAPIKeyError(request_id="req-1")
        assert err.code == 401
        assert err.error_code == "invalid_api_key"
        assert "API key" in err.message
        assert err.fix_suggestion is not None
        assert "rodiumai.io" in err.fix_suggestion

    def test_insufficient_rodi(self):
        err = InsufficientRODIError(request_id="req-2")
        assert err.code == 402
        assert err.error_code == "insufficient_rodi"
        assert "RODI" in err.message

    def test_permission_denied(self):
        err = PermissionDeniedError(request_id="req-3")
        assert err.code == 403
        assert err.error_code == "permission_denied"

    def test_model_not_found(self):
        err = ModelNotFoundError(request_id="req-4")
        assert err.code == 404
        assert err.error_code == "model_not_found"

    def test_rate_limit(self):
        err = RateLimitError(request_id="req-5", retry_after=2.5)
        assert err.code == 429
        assert err.error_code == "rate_limit_exceeded"
        assert err.retry_after == 2.5

    def test_internal_server_error(self):
        err = InternalServerError(request_id="req-6")
        assert err.code == 500
        assert err.error_code == "internal_error"

    def test_service_unavailable(self):
        err = ServiceUnavailableError(request_id="req-7")
        assert err.code == 503
        assert err.error_code == "service_unavailable"

    def test_timeout_error(self):
        err = TimeoutError_(elapsed=30.5, request_id="req-8")
        assert err.code == 408
        assert err.error_code == "timeout"
        assert "30.5" in err.message
        assert err.elapsed == 30.5

    def test_network_error(self):
        err = NetworkError("connection refused")
        assert err.code == 0
        assert err.error_code == "network_error"
        assert "connection" in err.message

    def test_all_errors_are_rodiumai_error(self):
        errors = [
            InvalidAPIKeyError(),
            InsufficientRODIError(),
            PermissionDeniedError(),
            ModelNotFoundError(),
            RateLimitError(),
            InternalServerError(),
            ServiceUnavailableError(),
            TimeoutError_(elapsed=1.0),
            NetworkError(),
        ]
        for err in errors:
            assert isinstance(err, RodiumAIError)

    def test_error_str_without_request_id(self):
        err = RodiumAIError("test error", code=400, error_code="custom_error")
        result = str(err)
        assert "[custom_error] test error" in result

    def test_error_str_with_request_id(self):
        err = RodiumAIError("test error", code=400, error_code="custom_error", request_id="req-abc")
        result = str(err)
        assert "[custom_error] test error" in result
        assert "req-abc" in result

    def test_error_str_on_subclass(self):
        err = InvalidAPIKeyError(request_id="req-1")
        result = str(err)
        assert "[invalid_api_key]" in result
        assert "req-1" in result

    def test_error_repr_contains_no_api_key(self):
        err = InvalidAPIKeyError()
        assert "rdk-" not in repr(err)

    def test_error_repr_format(self):
        err = InsufficientRODIError(request_id="req-xyz")
        assert "InsufficientRODIError" in repr(err)
        assert "402" in repr(err)
        assert "insufficient_rodi" in repr(err)

    def test_error_message_contains_fix_suggestion(self):
        err = InvalidAPIKeyError()
        assert err.fix_suggestion is not None

    def test_map_http_status_401(self):
        err = map_http_status(401, "req-1")
        assert isinstance(err, InvalidAPIKeyError)
        assert err.error_code == "invalid_api_key"

    def test_map_http_status_402(self):
        err = map_http_status(402, "req-2")
        assert isinstance(err, InsufficientRODIError)
        assert err.error_code == "insufficient_rodi"

    def test_map_http_status_403(self):
        err = map_http_status(403, "req-3")
        assert isinstance(err, PermissionDeniedError)
        assert err.error_code == "permission_denied"

    def test_map_http_status_404(self):
        err = map_http_status(404, "req-4")
        assert isinstance(err, ModelNotFoundError)
        assert err.error_code == "model_not_found"

    def test_map_http_status_429(self):
        err = map_http_status(429, "req-5")
        assert isinstance(err, RateLimitError)
        assert err.error_code == "rate_limit_exceeded"

    def test_map_http_status_500(self):
        err = map_http_status(500, "req-6")
        assert isinstance(err, InternalServerError)
        assert err.error_code == "internal_error"

    def test_map_http_status_503(self):
        err = map_http_status(503, "req-7")
        assert isinstance(err, ServiceUnavailableError)
        assert err.error_code == "service_unavailable"

    def test_map_http_status_unknown(self):
        err = map_http_status(418, "req-8")
        assert isinstance(err, RodiumAIError)
        assert err.error_code == "http_418"
