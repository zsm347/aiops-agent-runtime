from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.memory.embedding import (
    DeterministicEmbeddingService,
    OpenAICompatibleMemoryEmbeddingService,
    build_memory_embedding_service,
)
from superbiz_agent.memory.errors import (
    MemoryEmbeddingConfigurationError,
    MemoryEmbeddingContractError,
    MemoryEmbeddingUnavailableError,
)


@dataclass
class _EmbeddingItem:
    index: int
    embedding: list[float]


class _EmbeddingsEndpoint:
    def __init__(self, responses=(), *, error: Exception | None = None) -> None:
        self.responses = list(responses)
        self.error = error
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


class _Client:
    def __init__(self, responses=(), *, error: Exception | None = None) -> None:
        self.embeddings = _EmbeddingsEndpoint(responses, error=error)
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


def _response(*items: _EmbeddingItem, model: str = "text-embedding-v4"):
    return SimpleNamespace(model=model, data=list(items))


def _vector(value: float = 0.0) -> list[float]:
    return [value] * 1024


def _service(client: _Client, **overrides) -> OpenAICompatibleMemoryEmbeddingService:
    values = {
        "provider": "dashscope-openai-compatible",
        "model": "text-embedding-v4",
        "version": "text-embedding-v4-2026-07",
        "dimension": 1024,
        "batch_size": 2,
        "client": client,
    }
    values.update(overrides)
    return OpenAICompatibleMemoryEmbeddingService(**values)


@pytest.mark.asyncio
async def test_real_adapter_batches_documents_and_restores_provider_order() -> None:
    client = _Client(
        [
            _response(
                _EmbeddingItem(1, _vector(2.0)),
                _EmbeddingItem(0, _vector(1.0)),
            ),
            _response(_EmbeddingItem(0, _vector(3.0))),
        ]
    )
    service = _service(client)

    result = await service.embed_documents(["first", "second", "third"])

    assert [vector[0] for vector in result.vectors] == [1.0, 2.0, 3.0]
    assert result.identity.provider == "dashscope-openai-compatible"
    assert result.identity.model == "text-embedding-v4"
    assert result.identity.version == "text-embedding-v4-2026-07"
    assert result.identity.dimension == 1024
    assert client.embeddings.requests == [
        {
            "input": ["first", "second"],
            "model": "text-embedding-v4",
            "dimensions": 1024,
            "encoding_format": "float",
        },
        {
            "input": ["third"],
            "model": "text-embedding-v4",
            "dimensions": 1024,
            "encoding_format": "float",
        },
    ]


@pytest.mark.asyncio
async def test_real_adapter_has_distinct_query_and_document_entrypoints() -> None:
    client = _Client(
        [
            _response(_EmbeddingItem(0, _vector(1.0))),
            _response(_EmbeddingItem(0, _vector(2.0))),
        ]
    )
    service = _service(client)

    query = await service.embed_query("why did checkout fail")
    documents = await service.embed_documents(["checkout failed after pool exhaustion"])

    assert query.vector[0] == 1.0
    assert documents.vectors[0][0] == 2.0
    assert client.embeddings.requests[0]["input"] == ["why did checkout fail"]
    assert client.embeddings.requests[1]["input"] == [
        "checkout failed after pool exhaustion"
    ]


@pytest.mark.parametrize(
    "response",
    [
        _response(model="wrong-model"),
        _response(),
        _response(_EmbeddingItem(0, _vector()), _EmbeddingItem(0, _vector())),
        _response(_EmbeddingItem(2, _vector())),
        _response(_EmbeddingItem(0, [0.0] * 1023)),
        _response(_EmbeddingItem(0, [*([0.0] * 1023), float("nan")])),
        _response(_EmbeddingItem(0, [*([0.0] * 1023), True])),
    ],
)
@pytest.mark.asyncio
async def test_real_adapter_rejects_malformed_provider_contract(response) -> None:
    service = _service(_Client([response]))

    with pytest.raises(MemoryEmbeddingContractError):
        await service.embed_documents(["document"])


@pytest.mark.parametrize("value", [[], [""], ["   "], "not-a-sequence"])
@pytest.mark.asyncio
async def test_embedding_inputs_reject_blank_or_wrong_shape(value) -> None:
    service = _service(_Client([]))

    with pytest.raises(MemoryEmbeddingContractError):
        await service.embed_documents(value)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_provider_failure_is_typed_and_does_not_expose_details() -> None:
    secret = "private-key private-host private-content"
    service = _service(_Client(error=RuntimeError(secret)))

    with pytest.raises(MemoryEmbeddingUnavailableError) as exc_info:
        await service.embed_query("private-content")

    assert secret not in str(exc_info.value)
    assert "private" not in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "unknown"},
        {"model": ""},
        {"version": ""},
        {"dimension": 64},
        {"batch_size": 0},
        {"batch_size": 11},
        {"timeout_ms": 0},
    ],
)
def test_real_adapter_rejects_invalid_configuration(overrides: dict) -> None:
    with pytest.raises(MemoryEmbeddingConfigurationError):
        _service(_Client([]), **overrides)


def test_real_adapter_requires_independent_api_key_without_fake_client() -> None:
    with pytest.raises(MemoryEmbeddingConfigurationError):
        OpenAICompatibleMemoryEmbeddingService(
            provider="openai-compatible",
            model="text-embedding-3-large",
            version="text-embedding-3-large",
            dimension=1024,
            batch_size=10,
        )


@pytest.mark.asyncio
async def test_embedding_close_is_repeatable_and_closed_service_fails_safe() -> None:
    client = _Client([])
    service = _service(client)

    await service.aclose()
    await service.aclose()

    assert client.closed == 1
    with pytest.raises(MemoryEmbeddingUnavailableError):
        await service.embed_query("query")


@pytest.mark.asyncio
async def test_deterministic_adapter_implements_async_contract() -> None:
    service = DeterministicEmbeddingService(64)

    query = await service.embed_query("order service")
    documents = await service.embed_documents(["order service", "payment service"])

    assert query.identity.provider == "local-deterministic"
    assert len(query.vector) == 64
    assert documents.vectors[0] == query.vector
    await service.aclose()


def test_settings_keep_memory_embedding_credentials_independent() -> None:
    settings = Settings(
        _env_file=None,
        memory_embedding_provider="openai-compatible",
        memory_embedding_model="text-embedding-3-large",
        memory_embedding_version=None,
        memory_embedding_dimension=1024,
        memory_embedding_api_key="memory-secret",
        model_api_key="model-secret",
        rag_embedding_api_key="rag-secret",
    )

    assert settings.memory_embedding_version == "text-embedding-3-large"
    assert settings.memory_embedding_api_key == "memory-secret"
    assert "memory-secret" not in repr(settings)


def test_production_postgres_rejects_deterministic_embedding() -> None:
    with pytest.raises(ValueError, match="local-deterministic"):
        Settings(
            _env_file=None,
            app_env="production",
            memory_store_backend="postgres",
        )


def test_builder_uses_real_settings_without_model_or_rag_fallback() -> None:
    client = _Client([])
    settings = Settings(
        _env_file=None,
        memory_embedding_provider="openai-compatible",
        memory_embedding_model="text-embedding-3-large",
        memory_embedding_version="memory-v1",
        memory_embedding_dimension=1024,
        memory_embedding_batch_size=4,
        memory_embedding_api_key=None,
        model_api_key="must-not-be-used",
        rag_embedding_api_key="must-not-be-used",
    )

    service = build_memory_embedding_service(settings, client=client)

    assert service.identity.provider == "openai-compatible"
    assert service.identity.version == "memory-v1"
