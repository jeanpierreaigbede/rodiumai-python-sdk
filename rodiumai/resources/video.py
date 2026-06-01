from typing import Any, Optional


class Generations:
    async def create(
        self,
        *,
        model: str = "auto",
        prompt: str,
        duration: Optional[int] = None,
        resolution: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        raise NotImplementedError(
            "Video generation is not yet available. "
            "We are working on it and it will be released soon. "
            "Follow https://docs.rodiumai.io/changelog for updates."
        )


class Video:
    def __init__(self, http_client: Any):
        self.generations = Generations()
