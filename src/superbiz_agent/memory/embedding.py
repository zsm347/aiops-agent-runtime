from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol

from superbiz_agent.config import Settings
from superbiz_agent.memory.errors import (
    MemoryEmbeddingConfigurationError,
    MemoryEmbeddingContractError,
    MemoryEmbeddingUnavailableError,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-/]+|[\u4e00-\u9fff]+")
PRODUCTION_EMBEDDING_DIMENSION = 1024
REAL_MEMORY_EMBEDDING_PROVIDERS = frozenset(
    {"openai-compatible", "dashscope-openai-compatible"}
)


@dataclass(frozen=True)
class MemoryEmbeddingIdentity:
    provider: str
    model: str
    version: str
    dimension: int


@dataclass(frozen=True)
class MemoryEmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    identity: MemoryEmbeddingIdentity


@dataclass(frozen=True)
class MemoryQueryEmbedding:
    vector: tuple[float, ...]
    identity: MemoryEmbeddingIdentity


class MemoryEmbeddingService(Protocol):
    @property
    def identity(self) -> MemoryEmbeddingIdentity: ...

    async def embed_documents(self, texts: Sequence[str]) -> MemoryEmbeddingBatch: ...

    async def embed_query(self, text: str) -> MemoryQueryEmbedding: ...

    async def aclose(self) -> None: ...


class OpenAICompatibleMemoryEmbeddingService:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        version: str,
        dimension: int,
        batch_size: int,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_ms: int = 30_000,
        client: Any | None = None,
    ) -> None:
        if provider not in REAL_MEMORY_EMBEDDING_PROVIDERS:
            raise MemoryEmbeddingConfigurationError()
        if not _nonblank(model) or not _nonblank(version):
            raise MemoryEmbeddingConfigurationError()
        if dimension != PRODUCTION_EMBEDDING_DIMENSION:
            raise MemoryEmbeddingConfigurationError()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise MemoryEmbeddingConfigurationError()
        if provider == "dashscope-openai-compatible" and batch_size > 10:
            raise MemoryEmbeddingConfigurationError()
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms < 1:
            raise MemoryEmbeddingConfigurationError()
        if client is None and not _nonblank(api_key):
            raise MemoryEmbeddingConfigurationError()

        self._identity = MemoryEmbeddingIdentity(
            provider=provider,
            model=model.strip(),
            version=version.strip(),
            dimension=dimension,
        )
        self.batch_size = batch_size
        self._client = client or _build_openai_client(
            api_key=api_key,
            base_url=base_url,
            timeout_ms=timeout_ms,
        )
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def identity(self) -> MemoryEmbeddingIdentity:
        return self._identity

    async def embed_documents(self, texts: Sequence[str]) -> MemoryEmbeddingBatch:
        values = _validate_document_texts(texts)
        return MemoryEmbeddingBatch(
            vectors=await self._embed_values(values),
            identity=self.identity,
        )

    async def embed_query(self, text: str) -> MemoryQueryEmbedding:
        value = _validate_query_text(text)
        vectors = await self._embed_values((value,))
        return MemoryQueryEmbedding(vector=vectors[0], identity=self.identity)

    async def _embed_values(
        self,
        values: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        if self._closed:
            raise MemoryEmbeddingUnavailableError()
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(values), self.batch_size):
            request_batch = values[start : start + self.batch_size]
            try:
                response = await self._client.embeddings.create(
                    input=list(request_batch),
                    model=self.identity.model,
                    dimensions=self.identity.dimension,
                    encoding_format="float",
                )
            except Exception:
                raise MemoryEmbeddingUnavailableError() from None
            vectors.extend(_validate_provider_response(response, request_batch, self.identity))
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
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                raise MemoryEmbeddingUnavailableError() from None
        async with self._close_lock:
            self._closed = True


class DeterministicEmbeddingService:
    """Small local embedding used only for deterministic tests and evals."""

    def __init__(self, dimension: int = 64, *, version: str = "phase4-local") -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("dimension must be positive")
        if not _nonblank(version):
            raise ValueError("version must not be blank")
        self.dimension = dimension
        self._identity = MemoryEmbeddingIdentity(
            provider="local-deterministic",
            model="local-deterministic",
            version=version.strip(),
            dimension=dimension,
        )

    @property
    def identity(self) -> MemoryEmbeddingIdentity:
        return self._identity

    async def embed_documents(self, texts: Sequence[str]) -> MemoryEmbeddingBatch:
        values = _validate_document_texts(texts)
        return MemoryEmbeddingBatch(
            vectors=tuple(tuple(self.embed(text)) for text in values),
            identity=self.identity,
        )

    async def embed_query(self, text: str) -> MemoryQueryEmbedding:
        value = _validate_query_text(text)
        return MemoryQueryEmbedding(vector=tuple(self.embed(value)), identity=self.identity)

    async def aclose(self) -> None:
        return None

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokens(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            weight = 2.0 if _looks_like_slug_token(token) else 1.0
            vector[index] += weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        left_values = list(left)
        right_values = list(right)
        if not left_values or not right_values:
            return 0.0
        dot = sum(a * b for a, b in zip(left_values, right_values))
        left_norm = math.sqrt(sum(value * value for value in left_values))
        right_norm = math.sqrt(sum(value * value for value in right_values))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text or "")]


def build_memory_embedding_service(
    settings: Settings,
    *,
    client: Any | None = None,
) -> MemoryEmbeddingService:
    provider = settings.memory_embedding_provider.strip().lower()
    version = settings.memory_embedding_version or settings.memory_embedding_model
    if provider == "local-deterministic":
        return DeterministicEmbeddingService(
            settings.memory_embedding_dimension,
            version=version,
        )
    return OpenAICompatibleMemoryEmbeddingService(
        provider=provider,
        model=settings.memory_embedding_model,
        version=version,
        dimension=settings.memory_embedding_dimension,
        batch_size=settings.memory_embedding_batch_size,
        api_key=settings.memory_embedding_api_key,
        base_url=settings.memory_embedding_base_url,
        timeout_ms=settings.memory_embedding_timeout_ms,
        client=client,
    )


def _validate_document_texts(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise MemoryEmbeddingContractError()
    values = tuple(texts)
    if not values or any(not _nonblank(text) for text in values):
        raise MemoryEmbeddingContractError()
    return values


def _validate_query_text(text: str) -> str:
    if not _nonblank(text):
        raise MemoryEmbeddingContractError()
    return text


def _validate_provider_response(
    response: Any,
    request_batch: tuple[str, ...],
    identity: MemoryEmbeddingIdentity,
) -> tuple[tuple[float, ...], ...]:
    if getattr(response, "model", None) != identity.model:
        raise MemoryEmbeddingContractError()
    data = getattr(response, "data", None)
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        raise MemoryEmbeddingContractError()
    if len(data) != len(request_batch):
        raise MemoryEmbeddingContractError()
    ordered: list[tuple[float, ...] | None] = [None] * len(request_batch)
    for item in data:
        index = getattr(item, "index", None)
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(request_batch)
            or ordered[index] is not None
        ):
            raise MemoryEmbeddingContractError()
        ordered[index] = _validate_vector(getattr(item, "embedding", None), identity.dimension)
    if any(vector is None for vector in ordered):
        raise MemoryEmbeddingContractError()
    return tuple(vector for vector in ordered if vector is not None)


def _validate_vector(value: Any, dimension: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MemoryEmbeddingContractError()
    if len(value) != dimension:
        raise MemoryEmbeddingContractError()
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise MemoryEmbeddingContractError()
        number = float(item)
        if not math.isfinite(number):
            raise MemoryEmbeddingContractError()
        vector.append(number)
    return tuple(vector)


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _build_openai_client(
    *,
    api_key: str | None,
    base_url: str | None,
    timeout_ms: int,
) -> Any:
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError:
        raise MemoryEmbeddingConfigurationError() from None
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_ms / 1000,
        max_retries=0,
    )


def _looks_like_slug_token(token: str) -> bool:
    return any(ch.isascii() and (ch.isalnum() or ch in "-_/") for ch in token)
