import os
import re
from typing import Optional

from ._http import AsyncHTTPClient
from ._version import SDK_IDENTIFIER, VERSION
from .errors import InvalidAPIKeyError
from .logger import RodiumAILogger
from .resources import Audio, Chat, Embeddings, Images, Video
from .usage import UsageStats

_API_KEY_PATTERN = re.compile(r"^rdk-")


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
        self._api_key = api_key or os.environ.get("RODIUMAI_API_KEY", "")
        if not self._api_key:
            raise InvalidAPIKeyError()

        if not _API_KEY_PATTERN.match(self._api_key):
            self._logger = RodiumAILogger(log_level=log_level)
            self._logger.log_alert(
                "invalid_api_key_format",
                "API key format is invalid. Expected format: rdk-...",
                sdk_version=VERSION,
            )

        self._base_url = base_url
        self._timeout = timeout
        self._stream_timeout = stream_timeout
        self._max_retries = max_retries

        self._logger = RodiumAILogger(log_level=log_level)
        self._usage = UsageStats()

        self._http = AsyncHTTPClient(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            stream_timeout=stream_timeout,
            max_retries=max_retries,
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
