from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol

from superbiz_agent.config import Settings


class RagEmbeddingError(ValueError):
    """Raised when an embedding request or provider response violates the contract."""


@dataclass(frozen=True)
class RagEmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str
    version: str
    dimension: int


@dataclass(frozen=True)
class RagQueryEmbedding:
    vector: tuple[float, ...]
    provider: str
    model: str
    version: str
    dimension: int


@dataclass(frozen=True)
class RagEmbeddingIdentity:
    provider: str
    model: str
    version: str
    dimension: int


class RagEmbeddingService(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> RagEmbeddingBatch: ...

    async def embed_query(self, text: str) -> RagQueryEmbedding: ...


def _validate_texts(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise RagEmbeddingError("Embedding input must be a sequence of strings.")
    values = tuple(texts)
    if not values:
        raise RagEmbeddingError("Embedding input must not be empty.")
    if any(not isinstance(text, str) or not text.strip() for text in values):
        raise RagEmbeddingError("Embedding input must not contain blank text.")
    return values


def _validate_query_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise RagEmbeddingError("Embedding query must not be blank.")
    if len(text) > 2000:
        raise RagEmbeddingError("Embedding query must not exceed 2000 characters.")
    return text


def _validate_vector(vector: Any, dimension: int) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise RagEmbeddingError("Embedding vector must be a sequence of numbers.")
    if len(vector) != dimension:
        raise RagEmbeddingError(
            f"Embedding dimension mismatch: expected {dimension}, received {len(vector)}."
        )
    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RagEmbeddingError("Embedding vector contains a non-numeric value.")
        number = float(value)
        if not math.isfinite(number):
            raise RagEmbeddingError("Embedding vector contains a non-finite value.")
        values.append(number)
    return tuple(values)


def _validate_provider_batch(
    data: Sequence[Any], expected_count: int, dimension: int
) -> tuple[tuple[float, ...], ...]:
    if len(data) != expected_count:
        raise RagEmbeddingError(
            f"Embedding count mismatch: expected {expected_count}, received {len(data)}."
        )

    ordered: list[tuple[float, ...] | None] = [None] * expected_count
    for item in data:
        index = getattr(item, "index", None)
        if isinstance(index, bool) or not isinstance(index, int):
            raise RagEmbeddingError("Embedding provider returned an invalid index.")
        if index < 0 or index >= expected_count:
            raise RagEmbeddingError("Embedding provider returned an out-of-range index.")
        if ordered[index] is not None:
            raise RagEmbeddingError("Embedding provider returned a duplicate index.")
        ordered[index] = _validate_vector(getattr(item, "embedding", None), dimension)

    if any(vector is None for vector in ordered):
        raise RagEmbeddingError("Embedding provider response is missing an index.")
    return tuple(vector for vector in ordered if vector is not None)


def _response_data(response: Any) -> Sequence[Any]:
    data = getattr(response, "data", None)
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        raise RagEmbeddingError("Embedding provider returned invalid response data.")
    return data


class OpenAICompatibleRagEmbeddingService:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        version: str | None,
        dimension: int,
        batch_size: int = 32,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if client is None and not api_key:
            raise ValueError("An embedding API key is required.")
        self.provider = provider
        self.model = model
        self.version = version or model
        self.dimension = dimension
        self.batch_size = batch_size
        self._client = client or _build_openai_client(api_key=api_key, base_url=base_url)
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def embed_documents(self, texts: Sequence[str]) -> RagEmbeddingBatch:
        values = _validate_texts(texts)
        vectors = await self._embed_values(values)
        return RagEmbeddingBatch(
            vectors=vectors,
            provider=self.provider,
            model=self.model,
            version=self.version,
            dimension=self.dimension,
        )

    async def embed_query(self, text: str) -> RagQueryEmbedding:
        value = _validate_query_text(text)
        vectors = await self._embed_values((value,))
        return RagQueryEmbedding(
            vector=vectors[0],
            provider=self.provider,
            model=self.model,
            version=self.version,
            dimension=self.dimension,
        )

    async def _embed_values(self, values: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(values), self.batch_size):
            request_batch = values[start : start + self.batch_size]
            response = await self._client.embeddings.create(
                input=list(request_batch),
                model=self.model,
                dimensions=self.dimension,
            )
            try:
                vectors.extend(
                    _validate_provider_batch(
                        _response_data(response), len(request_batch), self.dimension
                    )
                )
            except RagEmbeddingError:
                raise
            except Exception:
                raise RagEmbeddingError(
                    "Embedding provider returned an invalid response contract."
                ) from None
        return tuple(vectors)

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            task = self._close_task
            if task is None or task.done():
                task = asyncio.create_task(self._close_client())
                self._close_task = task
        await asyncio.shield(task)

    async def _close_client(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
        async with self._close_lock:
            self._closed = True


class DeterministicRagEmbeddingService:
    """Stable local adapter for unit and Milvus Lite tests only."""

    def __init__(self, dimension: int = 64) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        self.dimension = dimension
        self.provider = "local-deterministic"
        self.model = "local-deterministic"
        self.version = "test-v1"

    async def embed_documents(self, texts: Sequence[str]) -> RagEmbeddingBatch:
        values = _validate_texts(texts)
        vectors = tuple(self._embed(text) for text in values)
        return RagEmbeddingBatch(
            vectors=vectors,
            provider=self.provider,
            model=self.model,
            version=self.version,
            dimension=self.dimension,
        )

    async def embed_query(self, text: str) -> RagQueryEmbedding:
        value = _validate_query_text(text)
        return RagQueryEmbedding(
            vector=self._embed(value),
            provider=self.provider,
            model=self.model,
            version=self.version,
            dimension=self.dimension,
        )

    def _embed(self, text: str) -> tuple[float, ...]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{counter}\0{text}".encode()).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        vector = values[: self.dimension]
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector)


def build_rag_embedding_service(
    settings: Settings, *, client: Any | None = None
) -> OpenAICompatibleRagEmbeddingService:
    api_key = settings.rag_embedding_api_key
    base_url = settings.rag_embedding_base_url
    if not api_key or not base_url:
        raise ValueError("RAG-specific embedding API key and base URL are required.")
    return OpenAICompatibleRagEmbeddingService(
        provider=settings.rag_embedding_provider,
        model=settings.rag_embedding_model,
        version=settings.rag_embedding_version,
        dimension=settings.rag_embedding_dimension,
        batch_size=settings.rag_embedding_batch_size,
        api_key=api_key,
        base_url=base_url,
        client=client,
    )


def _build_openai_client(*, api_key: str | None, base_url: str | None) -> Any:
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - base dependency in normal installs.
        raise RuntimeError("openai package is required for real RAG embedding") from exc
    return AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
