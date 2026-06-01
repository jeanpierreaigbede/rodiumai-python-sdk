import pytest

from rodiumai import RodiumAI
from rodiumai.errors import InvalidAPIKeyError


class TestRetryLogic:
    def test_no_retry_on_401(self):
        pass

    def test_no_retry_on_402(self):
        pass

    def test_no_retry_on_403(self):
        pass

    def test_no_retry_on_404(self):
        pass

    def test_retry_on_429(self):
        pass

    def test_retry_on_500(self):
        pass

    def test_exponential_backoff(self):
        pass

    def test_raises_final_error_after_retries_exhausted(self):
        pass
