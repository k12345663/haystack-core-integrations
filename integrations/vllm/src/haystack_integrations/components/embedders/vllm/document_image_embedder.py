# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace
from typing import Any

from haystack import Document, component, logging
from haystack.components.converters.image.image_utils import (
    _batch_convert_pdf_pages_to_images,
    _encode_image_to_base64,
    _extract_image_sources_info,
    _PDFPageInfo,
)
from haystack.dataclasses import ByteStream
from haystack.utils import Secret
from more_itertools import batched
from openai import APIError, AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse
from tqdm import tqdm

from haystack_integrations.common.vllm.utils import _create_async_openai_client, _create_openai_client

logger = logging.getLogger(__name__)


@component
class VLLMDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using multimodal models served with vLLM.

    It works with [vLLM](https://docs.vllm.ai/) servers running a multimodal embedding model.
    For each Document, the image (or PDF page) referenced in the `file_path_meta_field` metadata field is read,
    encoded as a base64 data URI and sent to vLLM's
    [chat embeddings API](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#multi-modal-inputs)
    (`/v1/embeddings` with a `messages` body). The resulting embedding is stored in the `embedding` field of the
    Document.

    vLLM's chat embeddings API returns one embedding per request. In `run`, requests are sent one after another.
    In `run_async`, up to `batch_size` requests are sent concurrently.

    ### Starting the vLLM server

    Start a vLLM server with a multimodal embedding model, for example:

    ```bash
    vllm serve openai/clip-vit-base-patch32 --runner pooling
    ```

    Some models need a custom chat template or extra server options. See the
    [vLLM multimodal embedding examples](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#multi-modal-inputs).

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.vllm import VLLMDocumentImageEmbedder

    embedder = VLLMDocumentImageEmbedder(model="openai/clip-vit-base-patch32")

    documents = [
        Document(content="A photo of a cat", meta={"file_path": "cat.jpg"}),
        Document(content="A scanned page", meta={"file_path": "report.pdf", "page_number": 1}),
    ]

    result = embedder.run(documents=documents)
    print(result["documents"][0].embedding)
    ```

    ### Usage example with a text instruction

    Some models (for example, VLM2Vec) expect a text instruction next to the image:

    ```python
    embedder = VLLMDocumentImageEmbedder(
        model="TIGER-Lab/VLM2Vec-Full",
        prompt="Represent the given image.",
    )
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
        api_base_url: str = "http://localhost:8000/v1",
        file_path_meta_field: str = "file_path",
        root_path: str | None = None,
        image_size: tuple[int, int] | None = None,
        prompt: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        raise_on_failure: bool = False,
        extra_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of VLLMDocumentImageEmbedder.

        :param model: The name of the multimodal embedding model served by vLLM. Check
            [vLLM documentation](https://docs.vllm.ai/en/stable/models/supported_models/) for supported models.
        :param api_key: The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
            Only required if the vLLM server was started with `--api-key`.
        :param api_base_url: The base URL of the vLLM server.
        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param image_size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time.
        :param prompt: Optional text sent together with each image in the same message, for models that expect
            an instruction (for example, `"Represent the given image."` for VLM2Vec).
        :param dimensions: The number of dimensions of the resulting embedding. Only models trained with
            Matryoshka Representation Learning support this parameter.
        :param batch_size: Maximum number of concurrent requests sent to vLLM in `run_async`.
            In `run`, requests are sent sequentially.
        :param progress_bar: Whether to show a progress bar.
        :param timeout: Timeout in seconds for vLLM client calls. If not set, the OpenAI client default applies.
        :param max_retries: Maximum number of retries for failed requests. If not set, the OpenAI client
            default applies.
        :param http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`. For more information, see the
            [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param raise_on_failure: Whether to raise an exception if an embedding request fails. If `False`,
            the component logs the error and leaves the embedding of the affected Document empty.
        :param extra_parameters: Additional parameters added to the request body of the vLLM chat embeddings
            endpoint, such as `continue_final_message`, `add_special_tokens` or `truncate_prompt_tokens`.
            See the [vLLM Embeddings API docs](https://docs.vllm.ai/en/stable/models/pooling_models/embed/).
        """
        if batch_size < 1:
            msg = "batch_size must be a positive integer."
            raise ValueError(msg)

        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.image_size = image_size
        self.prompt = prompt
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client_kwargs = http_client_kwargs
        self.raise_on_failure = raise_on_failure
        self.extra_parameters = extra_parameters

        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    def warm_up(self) -> None:
        """Create the synchronous OpenAI client."""
        if self._client is None:
            self._client = _create_openai_client(
                api_key=self.api_key,
                api_base_url=self.api_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client_kwargs=self.http_client_kwargs,
            )

    async def warm_up_async(self) -> None:
        """Create the asynchronous OpenAI client."""
        if self._async_client is None:
            self._async_client = _create_async_openai_client(
                api_key=self.api_key,
                api_base_url=self.api_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client_kwargs=self.http_client_kwargs,
            )

    def close(self) -> None:
        """Close the synchronous OpenAI client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close the asynchronous OpenAI client."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        if not isinstance(documents, list) or not all(isinstance(doc, Document) for doc in documents):
            msg = (
                "VLLMDocumentImageEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the VLLMTextEmbedder."
            )
            raise TypeError(msg)

    def _extract_images_to_embed(self, documents: list[Document]) -> list[str]:
        """
        Read the image or PDF page of each Document and return them as base64 data URIs.

        :raises ValueError: If a Document has no file path in `file_path_meta_field`, the file does not exist,
            or a PDF Document has no `page_number` in its metadata.
        :raises RuntimeError: If the conversion of some Documents fails.
        """
        images_source_info = _extract_image_sources_info(
            documents=documents, file_path_meta_field=self.file_path_meta_field, root_path=self.root_path
        )

        images_to_embed: list[str | None] = [None] * len(documents)
        pdf_page_infos: list[_PDFPageInfo] = []

        for doc_idx, image_source_info in enumerate(images_source_info):
            if image_source_info["mime_type"] == "application/pdf":
                # `_extract_image_sources_info` already raises if a PDF Document has no `page_number`
                pdf_page_infos.append(
                    {
                        "doc_idx": doc_idx,
                        "path": image_source_info["path"],
                        "page_number": image_source_info["page_number"],
                    }
                )
            else:
                image_byte_stream = ByteStream.from_file_path(
                    filepath=image_source_info["path"], mime_type=image_source_info["mime_type"]
                )
                mime_type, base64_image = _encode_image_to_base64(bytestream=image_byte_stream, size=self.image_size)
                images_to_embed[doc_idx] = f"data:{mime_type};base64,{base64_image}"

        base64_jpeg_images_by_doc_idx = _batch_convert_pdf_pages_to_images(
            pdf_page_infos=pdf_page_infos, return_base64=True, size=self.image_size
        )
        for doc_idx, base64_jpeg_image in base64_jpeg_images_by_doc_idx.items():
            images_to_embed[doc_idx] = f"data:image/jpeg;base64,{base64_jpeg_image}"

        failed_doc_ids = [documents[idx].id for idx, image in enumerate(images_to_embed) if image is None]
        if failed_doc_ids:
            msg = f"Conversion failed for some documents. Document IDs: {failed_doc_ids}."
            raise RuntimeError(msg)

        return [image for image in images_to_embed if image is not None]

    def _prepare_body(self, image_data_uri: str) -> dict[str, Any]:
        """Build the request body for vLLM's chat embeddings API for a single image."""
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": image_data_uri}}]
        if self.prompt:
            content.append({"type": "text", "text": self.prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            body["dimensions"] = self.dimensions
        if self.extra_parameters:
            body.update(self.extra_parameters)
        return body

    @staticmethod
    def _update_meta(meta: dict[str, Any], response: CreateEmbeddingResponse) -> None:
        if "model" not in meta:
            meta["model"] = response.model
        if response.usage is None:
            return
        if "usage" not in meta:
            meta["usage"] = dict(response.usage)
        else:
            meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
            meta["usage"]["total_tokens"] += response.usage.total_tokens

    def _handle_error(self, doc: Document, exc: APIError) -> None:
        logger.exception("Failed embedding of document {doc_id} caused by {exc}", doc_id=doc.id, exc=exc)
        if self.raise_on_failure:
            raise exc

    def _build_result(
        self, documents: list[Document], embeddings: list[list[float] | None], meta: dict[str, Any]
    ) -> dict[str, list[Document] | dict[str, Any]]:
        new_documents = []
        for doc, embedding in zip(documents, embeddings, strict=True):
            if embedding is None:
                new_documents.append(replace(doc))
                continue
            new_meta = {
                **doc.meta,
                "embedding_source": {"type": "image", "file_path_meta_field": self.file_path_meta_field},
            }
            new_documents.append(replace(doc, meta=new_meta, embedding=embedding))
        return {"documents": new_documents, "meta": meta}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Embed a list of image Documents.

        :param documents: Documents whose metadata contains the path to an image or a PDF file.
            PDF Documents must also have a `page_number` in their metadata.
        :returns: A dictionary with:
            - `documents`: The input documents with their `embedding` field populated.
            - `meta`: Information about the usage of the model.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": [], "meta": {}}

        images_to_embed = self._extract_images_to_embed(documents)

        self.warm_up()
        assert self._client is not None  # noqa: S101

        embeddings: list[list[float] | None] = []
        meta: dict[str, Any] = {}
        for doc, image in tqdm(
            zip(documents, images_to_embed, strict=True),
            total=len(documents),
            disable=not self.progress_bar,
            desc="Calculating embeddings",
        ):
            try:
                response = self._client.post(
                    "/embeddings", cast_to=CreateEmbeddingResponse, body=self._prepare_body(image)
                )
            except APIError as exc:
                self._handle_error(doc, exc)
                embeddings.append(None)
                continue
            embeddings.append(response.data[0].embedding)
            self._update_meta(meta, response)

        return self._build_result(documents, embeddings, meta)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Asynchronously embed a list of image Documents.

        Up to `batch_size` requests are sent to vLLM concurrently.

        :param documents: Documents whose metadata contains the path to an image or a PDF file.
            PDF Documents must also have a `page_number` in their metadata.
        :returns: A dictionary with:
            - `documents`: The input documents with their `embedding` field populated.
            - `meta`: Information about the usage of the model.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": [], "meta": {}}

        images_to_embed = self._extract_images_to_embed(documents)

        await self.warm_up_async()
        async_client = self._async_client
        assert async_client is not None  # noqa: S101

        embeddings: list[list[float] | None] = []
        meta: dict[str, Any] = {}
        with tqdm(total=len(documents), disable=not self.progress_bar, desc="Calculating embeddings") as pbar:
            for batch in batched(zip(documents, images_to_embed, strict=True), self.batch_size):
                responses = await asyncio.gather(
                    *(
                        async_client.post(
                            "/embeddings", cast_to=CreateEmbeddingResponse, body=self._prepare_body(image)
                        )
                        for _, image in batch
                    ),
                    return_exceptions=True,
                )
                for (doc, _), response in zip(batch, responses, strict=True):
                    if isinstance(response, APIError):
                        self._handle_error(doc, response)
                        embeddings.append(None)
                    elif isinstance(response, BaseException):
                        raise response
                    else:
                        embeddings.append(response.data[0].embedding)
                        self._update_meta(meta, response)
                pbar.update(len(batch))

        return self._build_result(documents, embeddings, meta)
