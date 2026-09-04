import copy
import os
import re
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union, cast

from ._http import AsyncHTTPClient
from ._version import VERSION
from .errors import InvalidAPIKeyError
from .logger import RodiumAILogger
from .resources import (
    Audio,
    EmbeddingsNamespace,
    Extensions,
    ImagesNamespace,
    MessagesNamespace,
    ModelsNamespace,
    Video,
)
from .resources.chat import ChatCompletion, ChatNamespace
from .usage import UsageStats

_API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9@._-]+$")
_MAX_RETRIES_LIMIT = 5
DEFAULT_MODEL = "openai/gpt-4o"


class RodiumAI:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.rodiumai.io/v1",
        timeout: float = 30.0,
        stream_timeout: float = 600.0,
        max_retries: int = 3,
        log_level: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        resolved_key = api_key or os.environ.get("RODIUMAI_API_KEY", "")

        if not resolved_key or not resolved_key.strip():
            raise InvalidAPIKeyError()

        if "\r" in resolved_key or "\n" in resolved_key or "\x00" in resolved_key:
            raise ValueError("API key contains invalid characters.")

        if not _API_KEY_PATTERN.match(resolved_key):
            self._logger = RodiumAILogger(log_level=log_level)
            self._logger.log_alert(
                "invalid_api_key_format",
                "API key format is invalid. Expected format: rd_sk_... or rdk-...",
                sdk_version=VERSION,
            )

        safe_retries = min(max_retries, _MAX_RETRIES_LIMIT)

        self._api_key = resolved_key
        self._base_url = base_url
        self._timeout = timeout
        self._stream_timeout = stream_timeout
        self._max_retries = safe_retries
        self._default_model = (
            default_model or os.environ.get("RODIUMAI_DEFAULT_MODEL") or DEFAULT_MODEL
        )

        self._pending_model: Optional[str] = None
        self._pending_temperature: Optional[float] = None
        self._pending_top_p: Optional[float] = None
        self._pending_max_tokens: Optional[int] = None
        self._pending_system_prompt: Optional[str] = None

        self._logger = RodiumAILogger(log_level=log_level)
        self._usage = UsageStats()

        self._http = AsyncHTTPClient(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            stream_timeout=stream_timeout,
            max_retries=safe_retries,
        )

        self.chat = ChatNamespace(self._http, self)
        self.embeddings = EmbeddingsNamespace(self._http, self)
        self.images = ImagesNamespace(self._http, self)
        self.audio = Audio(self._http)
        self.video = Video(self._http, self)
        self.models = ModelsNamespace(self._http)
        self.messages = MessagesNamespace(self._http)
        self._extensions = Extensions(self._http)

    @property
    def usage(self) -> UsageStats:
        return self._usage

    @property
    def logger(self) -> RodiumAILogger:
        return self._logger

    def _resolve_model(self) -> str:
        return self._pending_model or self._default_model

    def model(self, model: str) -> "RodiumAI":
        clone = self._clone()
        clone._pending_model = model
        return clone

    def temperature(self, temperature: float) -> "RodiumAI":
        clone = self._clone()
        clone._pending_temperature = temperature
        return clone

    def top_p(self, top_p: float) -> "RodiumAI":
        clone = self._clone()
        clone._pending_top_p = top_p
        return clone

    def max_tokens(self, max_tokens: int) -> "RodiumAI":
        clone = self._clone()
        clone._pending_max_tokens = max_tokens
        return clone

    def system_prompt(self, prompt: str) -> "RodiumAI":
        clone = self._clone()
        clone._pending_system_prompt = prompt
        return clone

    async def _flat_chat(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        **options: Any,
    ) -> ChatCompletion:
        built_messages = self._build_messages(messages)
        kwargs = self._chat_kwargs(options)
        result = await self.chat.completions.create(messages=built_messages, **kwargs)
        return cast(ChatCompletion, result)

    async def stream(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        **options: Any,
    ) -> AsyncIterator[str]:
        built_messages = self._build_messages(messages)
        kwargs = self._chat_kwargs({**options, "stream": True})
        stream_iter = await self.chat.completions.create(messages=built_messages, **kwargs)
        async for chunk in stream_iter:
            for choice in chunk.choices:
                if choice.delta.content:
                    yield choice.delta.content

    async def model_info(self, model_id: str, **options: Any) -> Dict[str, Any]:
        return await self.models.retrieve(model_id, timeout=options.get("timeout"))

    async def coding_models(self, **options: Any) -> Dict[str, Any]:
        return await self.models.list_coding(timeout=options.get("timeout"))

    async def videos(self, **options: Any) -> Any:
        return await self.video(**options)

    async def transcribe(self, file: Union[str, Path, bytes, Any], **options: Any) -> Any:
        model = options.pop("model", self._resolve_model())
        timeout = options.pop("timeout", None)
        return await self.audio.transcriptions.create(
            model=model,
            file=file,
            timeout=timeout,
            **options,
        )

    async def speech(self, **options: Any) -> bytes:
        model = options.pop("model", self._resolve_model())
        timeout = options.pop("timeout", None)
        result = await self.audio.speech.create(model=model, timeout=timeout, **options)
        return result.content

    async def wallet(self, **options: Any) -> Dict[str, Any]:
        return await self._extensions.wallet(timeout=options.get("timeout"))

    async def pricing(self, model: Optional[str] = None, **options: Any) -> Dict[str, Any]:
        return await self._extensions.pricing(model=model, timeout=options.get("timeout"))

    def _clone(self) -> "RodiumAI":
        return copy.copy(self)

    def _build_messages(
        self,
        messages: Union[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if isinstance(messages, str):
            built: List[Dict[str, Any]] = [{"role": "user", "content": messages}]
        else:
            built = list(messages)

        if self._pending_system_prompt:
            built = [{"role": "system", "content": self._pending_system_prompt}, *built]
        return built

    def _chat_kwargs(self, options: Dict[str, Any]) -> Dict[str, Any]:
        timeout = options.pop("timeout", None)
        model = options.pop("model", self._resolve_model())
        temperature = options.pop("temperature", self._pending_temperature)
        top_p = options.pop("top_p", self._pending_top_p)
        max_tokens = options.pop("max_tokens", self._pending_max_tokens)
        stream = options.pop("stream", False)

        kwargs: Dict[str, Any] = {"model": model, "stream": stream, **options}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _get_headers(self) -> dict[str, str]:
        return self._http._get_headers()

    def __repr__(self) -> str:
        masked = "rd_sk_****...****"
        return f"RodiumAI(api_key='{masked}', base_url='{self._base_url}')"

    def __str__(self) -> str:
        return self.__repr__()
