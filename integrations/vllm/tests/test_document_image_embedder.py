# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret
from openai import APIError
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from haystack_integrations.components.embedders.vllm import VLLMDocumentImageEmbedder

MODEL = "openai/clip-vit-base-patch32"
API_BASE_URL = "http://localhost:8003/v1"
TEST_FILES = Path(__file__).parent / "test_files"
IMAGE_PATH = str(TEST_FILES / "apple.jpg")
PDF_PATH = str(TEST_FILES / "sample_pdf_1.pdf")
TYPE = "haystack_integrations.components.embedders.vllm.document_image_embedder.VLLMDocumentImageEmbedder"


def _fake_response(embedding: list[float], prompt_tokens: int = 1) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        object="list",
        model="fake-model",
        data=[Embedding(object="embedding", index=0, embedding=embedding)],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


def _api_error() -> APIError:
    return APIError(message="boom", request=MagicMock(), body=None)


def _image_docs(n: int) -> list[Document]:
    return [Document(content=f"image {i}", meta={"file_path": IMAGE_PATH}) for i in range(n)]


class TestInitializationAndSerialization:
    def test_init_default(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        assert embedder.model == MODEL
        assert embedder.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert embedder.api_base_url == "http://localhost:8000/v1"
        assert embedder.file_path_meta_field == "file_path"
        assert embedder.root_path == ""
        assert embedder.image_size is None
        assert embedder.prompt is None
        assert embedder.dimensions is None
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.raise_on_failure is False
        assert embedder.extra_parameters is None
        assert embedder._client is None
        assert embedder._async_client is None

    def test_init_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            VLLMDocumentImageEmbedder(model=MODEL, batch_size=0)

    def test_to_dict_from_dict_round_trip(self):
        embedder = VLLMDocumentImageEmbedder(
            model="TIGER-Lab/VLM2Vec-Full",
            api_key=Secret.from_env_var("MY_VLLM_KEY", strict=False),
            api_base_url="http://my-vllm-server:8000/v1",
            file_path_meta_field="image_path",
            root_path="/images",
            image_size=(224, 224),
            prompt="Represent the given image.",
            dimensions=256,
            batch_size=4,
            progress_bar=False,
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"verify": False},
            raise_on_failure=True,
            extra_parameters={"truncate_prompt_tokens": 256},
        )
        data = component_to_dict(embedder, "embedder")
        assert data == {
            "type": TYPE,
            "init_parameters": {
                "model": "TIGER-Lab/VLM2Vec-Full",
                "api_key": {"env_vars": ["MY_VLLM_KEY"], "strict": False, "type": "env_var"},
                "api_base_url": "http://my-vllm-server:8000/v1",
                "file_path_meta_field": "image_path",
                "root_path": "/images",
                "image_size": (224, 224),
                "prompt": "Represent the given image.",
                "dimensions": 256,
                "batch_size": 4,
                "progress_bar": False,
                "timeout": 10.0,
                "max_retries": 2,
                "http_client_kwargs": {"verify": False},
                "raise_on_failure": True,
                "extra_parameters": {"truncate_prompt_tokens": 256},
            },
        }

        restored = component_from_dict(VLLMDocumentImageEmbedder, data, "embedder")
        assert component_to_dict(restored, "embedder") == data
        assert restored.api_key == Secret.from_env_var("MY_VLLM_KEY", strict=False)


class TestComponentLifecycle:
    @patch("haystack_integrations.components.embedders.vllm.document_image_embedder._create_openai_client")
    def test_warm_up_is_idempotent_and_close_resets(self, mock_create):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        embedder.warm_up()
        embedder.warm_up()
        mock_create.assert_called_once()

        client = embedder._client
        embedder.close()
        client.close.assert_called_once()
        assert embedder._client is None

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.embedders.vllm.document_image_embedder._create_async_openai_client")
    async def test_warm_up_async_is_idempotent_and_close_resets(self, mock_create):
        mock_create.return_value = MagicMock(close=AsyncMock())
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        await embedder.warm_up_async()
        await embedder.warm_up_async()
        mock_create.assert_called_once()

        client = embedder._async_client
        await embedder.close_async()
        client.close.assert_awaited_once()
        assert embedder._async_client is None


class TestHelpers:
    def test_prepare_body_minimal(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        assert embedder._prepare_body("data:image/jpeg;base64,abc") == {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}]}
            ],
            "encoding_format": "float",
        }

    def test_prepare_body_with_prompt_dimensions_and_extra_parameters(self):
        embedder = VLLMDocumentImageEmbedder(
            model=MODEL,
            prompt="Represent the given image.",
            dimensions=64,
            extra_parameters={"add_special_tokens": True},
        )
        body = embedder._prepare_body("data:image/jpeg;base64,abc")
        assert body["messages"][0]["content"][1] == {"type": "text", "text": "Represent the given image."}
        assert body["dimensions"] == 64
        assert body["add_special_tokens"] is True

    def test_extract_images_from_image_and_pdf(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, image_size=(64, 64))
        docs = [
            Document(content="apple", meta={"file_path": IMAGE_PATH}),
            Document(content="pdf page", meta={"file_path": PDF_PATH, "page_number": 1}),
        ]
        images = embedder._extract_images_to_embed(docs)
        assert len(images) == 2
        assert images[0].startswith("data:image/jpeg;base64,")
        assert images[1].startswith("data:image/jpeg;base64,")

    def test_extract_images_pdf_without_page_number(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        with pytest.raises(ValueError, match="missing the 'page_number' key"):
            embedder._extract_images_to_embed([Document(content="pdf", meta={"file_path": PDF_PATH})])

    def test_extract_images_with_root_path(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, root_path=str(TEST_FILES))
        images = embedder._extract_images_to_embed([Document(content="apple", meta={"file_path": "apple.jpg"})])
        assert images[0].startswith("data:image/jpeg;base64,")


class TestRun:
    @pytest.mark.parametrize("documents", ["text", [1, 2, 3]])
    def test_run_wrong_input_format(self, documents):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        with pytest.raises(TypeError, match=r"VLLMDocumentImageEmbedder expects a list of Documents as input\."):
            embedder.run(documents=documents)

    def test_run_empty(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}

    def test_run(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.post.side_effect = [_fake_response([0.1], 2), _fake_response([0.2], 3)]

        docs = _image_docs(2)
        result = embedder.run(docs)

        assert [d.embedding for d in result["documents"]] == [[0.1], [0.2]]
        assert result["documents"][0].meta["embedding_source"] == {"type": "image", "file_path_meta_field": "file_path"}
        assert result["meta"] == {"model": "fake-model", "usage": {"prompt_tokens": 5, "total_tokens": 5}}
        # input documents are not mutated
        assert docs[0].embedding is None
        assert "embedding_source" not in docs[0].meta

        call = embedder._client.post.call_args_list[0]
        assert call.args == ("/embeddings",)
        assert call.kwargs["cast_to"] is CreateEmbeddingResponse
        assert call.kwargs["body"]["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg")

    def test_run_continues_on_api_error(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.post.side_effect = [_api_error(), _fake_response([0.2])]

        result = embedder.run(_image_docs(2))

        assert result["documents"][0].embedding is None
        assert "embedding_source" not in result["documents"][0].meta
        assert result["documents"][1].embedding == [0.2]

    def test_run_raise_on_failure(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, raise_on_failure=True, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.post.side_effect = _api_error()

        with pytest.raises(APIError):
            embedder.run(_image_docs(1))


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_run_async_empty(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        assert await embedder.run_async(documents=[]) == {"documents": [], "meta": {}}

    @pytest.mark.asyncio
    async def test_run_async_batches_concurrent_requests(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, batch_size=2, progress_bar=False)
        embedder._async_client = MagicMock()
        embedder._async_client.post = AsyncMock(
            side_effect=[_fake_response([0.1]), _fake_response([0.2]), _fake_response([0.3])]
        )

        result = await embedder.run_async(_image_docs(3))

        assert [d.embedding for d in result["documents"]] == [[0.1], [0.2], [0.3]]
        assert embedder._async_client.post.await_count == 3
        assert result["meta"] == {"model": "fake-model", "usage": {"prompt_tokens": 3, "total_tokens": 3}}

    @pytest.mark.asyncio
    async def test_run_async_continues_on_api_error(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, progress_bar=False)
        embedder._async_client = MagicMock()
        embedder._async_client.post = AsyncMock(side_effect=[_fake_response([0.1]), _api_error()])

        result = await embedder.run_async(_image_docs(2))

        assert result["documents"][0].embedding == [0.1]
        assert result["documents"][1].embedding is None

    @pytest.mark.asyncio
    async def test_run_async_raise_on_failure(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, raise_on_failure=True, progress_bar=False)
        embedder._async_client = MagicMock()
        embedder._async_client.post = AsyncMock(side_effect=_api_error())

        with pytest.raises(APIError):
            await embedder.run_async(_image_docs(1))


class TestIntegration:
    @pytest.mark.integration
    def test_live_run(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, api_base_url=API_BASE_URL)
        docs = [
            Document(content="apple", meta={"file_path": IMAGE_PATH}),
            Document(content="pdf page", meta={"file_path": PDF_PATH, "page_number": 1}),
        ]

        result = embedder.run(docs)

        assert len(result["documents"]) == 2
        for doc in result["documents"]:
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)
            assert doc.meta["embedding_source"]["type"] == "image"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, api_base_url=API_BASE_URL)
        docs = [
            Document(content="apple", meta={"file_path": IMAGE_PATH}),
            Document(content="pdf page", meta={"file_path": PDF_PATH, "page_number": 1}),
        ]

        result = await embedder.run_async(docs)

        assert len(result["documents"]) == 2
        for doc in result["documents"]:
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)
