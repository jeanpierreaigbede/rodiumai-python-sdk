from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

from .._http import AsyncHTTPClient


@dataclass
class Transcription:
    text: str = ""


@dataclass
class SpeechResponse:
    content: bytes = b""
    content_type: str = "audio/mpeg"


class Transcriptions:
    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str = "auto",
        file: Any,
        language: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Transcription:
        files = {"file": file}
        data: Dict[str, Any] = {"model": model}
        if language is not None:
            data["language"] = language

        status_code, response_data, request_id, error = await self._http._request(
            "POST", "/audio/transcriptions", json_body=data, files=files, timeout=timeout
        )
        if error:
            raise error

        return Transcription(text=response_data.get("text", ""))


class Speech:
    def __init__(self, http_client: AsyncHTTPClient):
        self._http = http_client

    async def create(
        self,
        *,
        model: str = "auto",
        input: str,
        voice: str = "alloy",
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> SpeechResponse:
        body: Dict[str, Any] = {
            "model": model,
            "input": input,
            "voice": voice,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if speed is not None:
            body["speed"] = speed

        status_code, data, request_id, error = await self._http._request(
            "POST", "/audio/speech", json_body=body, timeout=timeout
        )
        if error:
            raise error

        content = data.get("content", b"")
        if isinstance(content, str):
            content = content.encode("utf-8")

        return SpeechResponse(
            content=content,
            content_type=data.get("content_type", "audio/mpeg"),
        )


class Audio:
    def __init__(self, http_client: AsyncHTTPClient):
        self.transcriptions = Transcriptions(http_client)
        self.speech = Speech(http_client)
