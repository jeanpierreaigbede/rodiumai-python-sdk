import os
import re
from typing import Optional

from ._http import AsyncHTTPClient
from ._version import SDK_IDENTIFIER, VERSION
from .errors import InvalidAPIKeyError
from .logger import RodiumAILogger
from .resources import Audio, Chat, Embeddings, Images, Video
from .usage import UsageStats

_API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9@._-]+$")
_MAX_RETRIES_LIMIT = 5


class RodiumAI:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.rodiumai.io/v1",
        timeout: float = 30.0,
        stream_timeout: float = 600.0,
        max_retries: int = 3,
        log_level: Optional[str] = None,
    ):
        resolved_key = api_key or os.environ.get("RODIUMAI_API_KEY", "")

        if not resolved_key or not resolved_key.strip():
            raise ValueError("API key must not be empty. Provide a valid RodiumAI API key.")

        # Reject header injection characters
        if "\r" in resolved_key or "\n" in resolved_key or "\x00" in resolved_key:
            raise ValueError("API key contains invalid characters.")

        if not _API_KEY_PATTERN.match(resolved_key):
            self._logger = RodiumAILogger(log_level=log_level)
            self._logger.log_alert(
                "invalid_api_key_format",
                "API key format is invalid. Expected format: rdk-... or alphanumeric token",
                sdk_version=VERSION,
            )

        # Cap max_retries to prevent DoS-style abuse
        safe_retries = min(max_retries, _MAX_RETRIES_LIMIT)

        self._api_key = resolved_key
        self._base_url = base_url
        self._timeout = timeout
        self._stream_timeout = stream_timeout
        self._max_retries = safe_retries

        self._logger = RodiumAILogger(log_level=log_level)
        self._usage = UsageStats()

        self._http = AsyncHTTPClient(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            stream_timeout=stream_timeout,
            max_retries=safe_retries,
        )

        self.chat = Chat(self._http)
        self.embeddings = Embeddings(self._http)
        self.images = Images(self._http)
        self.audio = Audio(self._http)
        self.video = Video(self._http)

    @property
    def usage(self) -> UsageStats:
        return self._usage

    @property
    def logger(self) -> RodiumAILogger:
        return self._logger

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "X-RodiumAI-SDK": SDK_IDENTIFIER,
            "X-RodiumAI-Version": VERSION,
            "Content-Type": "application/json",
        }

    def __repr__(self) -> str:
        masked = "rdk-****...****"
        return f"RodiumAI(api_key='{masked}', base_url='{self._base_url}')"

    def __str__(self) -> str:
        return self.__repr__()
