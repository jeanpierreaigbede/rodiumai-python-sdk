from typing import Optional


class RodiumAIError(Exception):
    def __init__(
        self,
        message: str,
        code: int = 0,
        request_id: Optional[str] = None,
        fix_suggestion: Optional[str] = None,
        docs_url: Optional[str] = None,
    ):
        self.message = message
        self.code = code
        self.request_id = request_id
        self.fix_suggestion = fix_suggestion
        self.docs_url = docs_url
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code}, request_id={self.request_id})"


class InvalidAPIKeyError(RodiumAIError):
    def __init__(self, request_id: Optional[str] = None):
        super().__init__(
            message="Invalid or missing API key. Provide a valid RodiumAI API key.",
            code=401,
            request_id=request_id,
            fix_suggestion="Check your API key at https://rodiumai.io/dashboard",
            docs_url="https://docs.rodiumai.io/api-keys",
        )


class InsufficientRODIError(RodiumAIError):
    def __init__(self, request_id: Optional[str] = None):
        super().__init__(
            message="You do not have enough RODI credits to complete this request.",
            code=402,
            request_id=request_id,
            fix_suggestion="Top up your wallet at https://rodiumai.io/wallet",
            docs_url="https://docs.rodiumai.io/billing",
        )


class PermissionDeniedError(RodiumAIError):
    def __init__(self, request_id: Optional[str] = None):
        super().__init__(
            message="You do not have permission to perform this action.",
            code=403,
            request_id=request_id,
            fix_suggestion="Verify your API key has the required permissions at https://rodiumai.io/dashboard",
            docs_url="https://docs.rodiumai.io/permissions",
        )


class ModelNotFoundError(RodiumAIError):
    def __init__(self, request_id: Optional[str] = None):
        super().__init__(
            message="The requested model was not found or is not available.",
            code=404,
            request_id=request_id,
            fix_suggestion="Check available models at https://docs.rodiumai.io/models",
            docs_url="https://docs.rodiumai.io/models",
        )


class RateLimitError(RodiumAIError):
    def __init__(
        self,
        request_id: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        self.retry_after = retry_after
        msg = "Rate limit exceeded. Too many requests."
        super().__init__(
            message=msg,
            code=429,
            request_id=request_id,
            fix_suggestion="Retry after the suggested delay. Consider upgrading your plan at https://rodiumai.io/pricing",
            docs_url="https://docs.rodiumai.io/rate-limits",
        )


class InternalServerError(RodiumAIError):
    def __init__(self, request_id: Optional[str] = None):
        super().__init__(
            message="Internal server error. Our team has been notified.",
            code=500,
            request_id=request_id,
            fix_suggestion="Retry your request. If the problem persists, contact support at https://rodiumai.io/support",
            docs_url="https://docs.rodiumai.io/troubleshooting",
        )


class ServiceUnavailableError(RodiumAIError):
    def __init__(self, request_id: Optional[str] = None):
        super().__init__(
            message="Service is temporarily unavailable. Please try again later.",
            code=503,
            request_id=request_id,
            fix_suggestion="Retry after a few seconds. Check https://status.rodiumai.io for outages.",
            docs_url="https://status.rodiumai.io",
        )


class TimeoutError_(RodiumAIError):
    def __init__(
        self,
        elapsed: float,
        request_id: Optional[str] = None,
    ):
        self.elapsed = elapsed
        super().__init__(
            message=f"Request timed out after {elapsed:.1f}s.",
            code=408,
            request_id=request_id,
            fix_suggestion="Check your network connection or increase the timeout.",
            docs_url="https://docs.rodiumai.io/timeouts",
        )


class NetworkError(RodiumAIError):
    def __init__(self, message: str = "Network connection failed."):
        super().__init__(
            message=message,
            code=0,
            fix_suggestion="Check your internet connection and firewall settings.",
            docs_url="https://docs.rodiumai.io/troubleshooting",
        )


HTTP_STATUS_TO_ERROR: dict[int, type[RodiumAIError]] = {
    401: InvalidAPIKeyError,
    402: InsufficientRODIError,
    403: PermissionDeniedError,
    404: ModelNotFoundError,
    429: RateLimitError,
    500: InternalServerError,
    503: ServiceUnavailableError,
}


def map_http_status(status_code: int, request_id: Optional[str] = None) -> RodiumAIError:
    error_cls = HTTP_STATUS_TO_ERROR.get(status_code, RodiumAIError)
    if error_cls is RateLimitError:
        return error_cls(request_id=request_id)
    return error_cls(request_id=request_id)
