import pytest

from rodiumai.resources.embeddings import Embedding, EmbeddingsResponse


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_single_text_returns_vector(self, httpx_mock, client, mock_embedding_response):
        httpx_mock.add_response(
            url="https://api.rodiumai.io/v1/embeddings",
            method="POST",
            json=mock_embedding_response,
        )
        response = await client.embeddings.create(model="auto", input="Hello")
        assert isinstance(response, EmbeddingsResponse)
        assert isinstance(response.data[0], Embedding)
        assert isinstance(response.data[0].embedding, list)
        assert all(isinstance(x, float) for x in response.data[0].embedding)
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
