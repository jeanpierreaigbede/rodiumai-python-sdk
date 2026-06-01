import pytest

from rodiumai.resources.images import ImagesResponse


class TestImagesIntegration:
    @pytest.mark.asyncio
    async def test_images_generations_endpoint(self, httpx_mock, client, mock_image_response):
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
        assert response.data[0].url is not None

    @pytest.mark.asyncio
    async def test_response_structure(self, httpx_mock, client, mock_image_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/images/generations",
            method="POST",
            json=mock_image_response,
        )
        response = await client.images.generate(model="auto", prompt="test")
        assert response.created > 0
        assert len(response.data) == 1
