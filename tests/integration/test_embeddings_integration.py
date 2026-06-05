import pytest

from rodiumai.resources.embeddings import EmbeddingsResponse


class TestEmbeddingsIntegration:
    @pytest.mark.asyncio
    async def test_single_text_input(self, httpx_mock, client, mock_embedding_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/embeddings",
            method="POST",
            json=mock_embedding_response,
        )
        response = await client.embeddings.create(model="auto", input="Hello")
        assert isinstance(response, EmbeddingsResponse)
        assert len(response.data) == 1
        assert len(response.data[0].embedding) == 3

    @pytest.mark.asyncio
    async def test_batch_input(self, httpx_mock, client, mock_batch_embedding_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/embeddings",
            method="POST",
            json=mock_batch_embedding_response,
        )
        response = await client.embeddings.create(
            model="auto",
            input=["Hello", "World"],
        )
        assert len(response.data) == 2
        assert response.data[0].index == 0
        assert response.data[1].index == 1

    @pytest.mark.asyncio
    async def test_response_structure(self, httpx_mock, client, mock_embedding_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/embeddings",
            method="POST",
            json=mock_embedding_response,
        )
        response = await client.embeddings.create(model="auto", input="test")
        assert response.object == "list"
        assert response.model == "auto"
        assert response.usage.total_tokens == 5
