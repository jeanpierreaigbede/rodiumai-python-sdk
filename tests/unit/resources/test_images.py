import pytest

from rodiumai.resources.images import ImageData, ImagesResponse


class TestImages:
    @pytest.mark.asyncio
    async def test_generate_returns_image_data(self, httpx_mock, client, mock_image_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/images/generations",
            method="POST",
            json=mock_image_response,
        )
        response = await client.images.generate(
            model="auto",
            prompt="A sunset",
            n=1,
            size="1024x1024",
        )
        assert isinstance(response, ImagesResponse)
        assert isinstance(response.data[0], ImageData)
        assert response.data[0].url is not None
        assert "generated" in response.data[0].url
