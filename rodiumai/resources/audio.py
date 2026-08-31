from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .._http import AsyncHTTPClient


@dataclass
class Transcription:
    text: str = ""


@dataclass
class SpeechResponse:
    content: bytes = b""
    content_type: str = "audio/mpeg"


class Transcriptions:
    DEFAULT_MODEL = "openai/gpt-4o"

    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str = DEFAULT_MODEL,
        file: Any,
        language: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Transcription:
        upload = self._prepare_file(file)
        data: Dict[str, Any] = {"model": model, **kwargs}
        if language is not None:
            data["language"] = language
        data = {k: v for k, v in data.items() if v is not None}

        _, response_data, _, error = await self._http._request(
            "POST",
            "/audio/transcriptions",
            files={"file": upload},
            data=data,
            timeout=timeout,
        )
        if error:
            raise error

        return Transcription(text=response_data.get("text", ""))

    @staticmethod
    def _prepare_file(file: Any) -> Any:
        if isinstance(file, (str, Path)):
            path = Path(file)
            return (path.name, path.read_bytes())
        if isinstance(file, bytes):
            return ("audio.bin", file)
        return file


class Speech:
    DEFAULT_MODEL = "openai/gpt-4o"

    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str = DEFAULT_MODEL,
        input: str,
        voice: str = "alloy",
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> SpeechResponse:
        body: Dict[str, Any] = {
            "model": model,
            "input": input,
            "voice": voice,
            **kwargs,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if speed is not None:
            body["speed"] = speed

        content, content_type, error = await self._http._request_binary(
            "POST", "/audio/speech", json_body=body, timeout=timeout
        )
        if error:
            raise error

        return SpeechResponse(content=content, content_type=content_type)


class Audio:
    def __init__(self, http_client: AsyncHTTPClient):
        self.transcriptions = Transcriptions(http_client)
        self.speech = Speech(http_client)
