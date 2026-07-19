from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.rag.embedding import (
    DeterministicRagEmbeddingService,
    OpenAICompatibleRagEmbeddingService,
    RagEmbeddingError,
    RagQueryEmbedding,
    build_rag_embedding_service,
)


class _Embeddings:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=self.responses.pop(0))


def _item(index: int, vector: list[float]):
    return SimpleNamespace(index=index, embedding=vector)


@pytest.mark.asyncio
async def test_openai_adapter_batches_requests_and_reorders_each_provider_batch() -> None:
    embeddings = _Embeddings(
        [
            [_item(1, [0.0, 1.0]), _item(0, [1.0, 0.0])],
            [_item(0, [0.5, 0.5])],
        ]
    )
    service = OpenAICompatibleRagEmbeddingService(
        provider="compatible",
        model="text-embedding-v4",
        version="version-4",
        dimension=2,
        batch_size=2,
        client=SimpleNamespace(embeddings=embeddings),
    )

    result = await service.embed_documents(["first", "second", "third"])

    assert result.vectors == ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))
    assert result.provider == "compatible"
    assert result.version == "version-4"
    assert embeddings.calls == [
        {
            "input": ["first", "second"],
            "model": "text-embedding-v4",
            "dimensions": 2,
        },
        {"input": ["third"], "model": "text-embedding-v4", "dimensions": 2},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        [_item(0, [1.0, 0.0])],
        [_item(0, [1.0, 0.0]), _item(0, [0.0, 1.0])],
        [_item(0, [1.0]), _item(1, [0.0, 1.0])],
        [_item(0, [float("nan"), 0.0]), _item(1, [0.0, 1.0])],
        [_item(0, [float("inf"), 0.0]), _item(1, [0.0, 1.0])],
    ],
)
async def test_openai_adapter_rejects_invalid_provider_shape(response) -> None:
    service = OpenAICompatibleRagEmbeddingService(
        provider="compatible",
        model="model",
        version=None,
        dimension=2,
        client=SimpleNamespace(embeddings=_Embeddings([response])),
    )
    with pytest.raises(RagEmbeddingError):
        await service.embed_documents(["first", "second"])


class _RawEmbeddings:
    def __init__(self, response):
        self.response = response

    async def create(self, **kwargs):
        del kwargs
        return self.response


class _ExplodingEmbeddingResponse:
    @property
    def data(self):
        raise AttributeError("malformed provider property")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        object(),
        SimpleNamespace(data=None),
        SimpleNamespace(data="not-a-data-sequence"),
        SimpleNamespace(data=b"not-a-data-sequence"),
        SimpleNamespace(data=42),
    ],
)
async def test_openai_adapter_normalizes_invalid_top_level_response_data(response) -> None:
    service = OpenAICompatibleRagEmbeddingService(
        provider="compatible",
        model="model",
        version=None,
        dimension=2,
        client=SimpleNamespace(embeddings=_RawEmbeddings(response)),
    )

    with pytest.raises(RagEmbeddingError, match="response data"):
        await service.embed_documents(["first"])


@pytest.mark.asyncio
async def test_openai_adapter_maps_malformed_response_properties_to_contract_error() -> None:
    service = OpenAICompatibleRagEmbeddingService(
        provider="compatible",
        model="model",
        version="v1",
        dimension=2,
        client=SimpleNamespace(embeddings=_RawEmbeddings(_ExplodingEmbeddingResponse())),
    )
    with pytest.raises(RagEmbeddingError, match="response"):
        await service.embed_documents(["first"])


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [[], [""], ["  "]])
async def test_embedding_input_must_be_nonempty_and_nonblank(texts) -> None:
    service = DeterministicRagEmbeddingService(dimension=4)
    with pytest.raises(RagEmbeddingError):
        await service.embed_documents(texts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "texts",
    ["single string", b"bytes", {"mapping": "value"}, {"set-value"}],
)
async def test_document_embedding_rejects_non_sequence_and_scalar_containers(texts) -> None:
    service = DeterministicRagEmbeddingService(dimension=4)
    with pytest.raises(RagEmbeddingError, match="sequence of strings"):
        await service.embed_documents(texts)


@pytest.mark.asyncio
async def test_document_embedding_rejects_generator_input() -> None:
    service = DeterministicRagEmbeddingService(dimension=4)
    with pytest.raises(RagEmbeddingError, match="sequence of strings"):
        await service.embed_documents(text for text in ("first", "second"))


@pytest.mark.asyncio
async def test_deterministic_adapter_is_stable_and_factory_remains_production_adapter() -> None:
    deterministic = DeterministicRagEmbeddingService(dimension=4)
    first = await deterministic.embed_documents(["中文 runbook"])
    second = await deterministic.embed_documents(["中文 runbook"])
    assert first == second
    assert len(first.vectors[0]) == 4

    settings = Settings(
        _env_file=None,
        rag_embedding_api_key="rag-key",
        model_api_key="model-key",
        rag_embedding_base_url="https://rag.example/v1",
        model_base_url="https://model.example/v1",
        rag_embedding_version=None,
    )
    service = build_rag_embedding_service(
        settings, client=SimpleNamespace(embeddings=_Embeddings([]))
    )
    assert isinstance(service, OpenAICompatibleRagEmbeddingService)
    assert service.version == settings.rag_embedding_model


def test_rag_chunk_config_and_embedding_version_are_validated() -> None:
    settings = Settings(_env_file=None, rag_embedding_version=None)
    assert settings.rag_embedding_version == settings.rag_embedding_model
    with pytest.raises(ValueError, match="overlap"):
        Settings(_env_file=None, rag_chunk_size_tokens=10, rag_chunk_overlap_tokens=10)
    with pytest.raises(ValueError, match="less than or equal to 1000"):
        Settings(_env_file=None, rag_hybrid_top_k=1001)


def test_embedding_factory_does_not_fall_back_to_model_gateway_credentials() -> None:
    settings = Settings(
        _env_file=None,
        model_api_key="model-only-key",
        model_base_url="https://model.invalid/v1",
        rag_embedding_api_key=None,
        rag_embedding_base_url=None,
    )
    with pytest.raises(ValueError, match="RAG-specific"):
        build_rag_embedding_service(settings)


def test_settings_repr_and_str_do_not_expose_embedding_credentials() -> None:
    secrets = {
        "model_api_key": "secret-model-fallback-7e934d",
        "rag_milvus_token": "secret-milvus-token-42ab91",
        "rag_embedding_api_key": "secret-embedding-key-c5d128",
        "rag_rerank_api_key": "secret-rerank-key-a8110f",
    }
    settings = Settings(_env_file=None, **secrets)

    rendered = (repr(settings), str(settings))
    for field_name, secret in secrets.items():
        assert secret not in rendered[0]
        assert secret not in rendered[1]
        assert getattr(settings, field_name) == secret


@pytest.mark.asyncio
async def test_openai_query_embedding_reuses_request_and_validation_contract() -> None:
    embeddings = _Embeddings([[_item(0, [0.25, 0.75])]])
    service = OpenAICompatibleRagEmbeddingService(
        provider="compatible",
        model="text-embedding-v4",
        version="version-4",
        dimension=2,
        client=SimpleNamespace(embeddings=embeddings),
    )

    result = await service.embed_query("中文订单超时 runbook")

    assert result == RagQueryEmbedding(
        vector=(0.25, 0.75),
        provider="compatible",
        model="text-embedding-v4",
        version="version-4",
        dimension=2,
    )
    assert embeddings.calls == [
        {
            "input": ["中文订单超时 runbook"],
            "model": "text-embedding-v4",
            "dimensions": 2,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "  ", "x" * 2001])
async def test_query_embedding_rejects_blank_and_oversized_text(text: str) -> None:
    service = DeterministicRagEmbeddingService(dimension=4)
    with pytest.raises(RagEmbeddingError):
        await service.embed_query(text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector",
    [
        [1.0],
        [True, 0.0],
        [float("nan"), 0.0],
        [float("inf"), 0.0],
    ],
)
async def test_query_embedding_rejects_invalid_provider_vector(vector: list[float]) -> None:
    service = OpenAICompatibleRagEmbeddingService(
        provider="compatible",
        model="model",
        version="v1",
        dimension=2,
        client=SimpleNamespace(embeddings=_Embeddings([[_item(0, vector)]])),
    )
    with pytest.raises(RagEmbeddingError):
        await service.embed_query("query")


@pytest.mark.asyncio
async def test_deterministic_query_embedding_is_stable_and_semantically_typed() -> None:
    service = DeterministicRagEmbeddingService(dimension=8)
    first = await service.embed_query("中文 runbook")
    second = await service.embed_query("中文 runbook")

    assert first == second
    assert len(first.vector) == first.dimension == 8
    assert all(math.isfinite(value) for value in first.vector)


def test_production_embedding_client_disables_openai_sdk_retries(monkeypatch) -> None:
    captured = {}
    fake_client = SimpleNamespace(embeddings=_Embeddings([]))

    def client_factory(**kwargs):
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr("openai.AsyncOpenAI", client_factory)
    service = OpenAICompatibleRagEmbeddingService(
        provider="compatible",
        model="model",
        version="v1",
        dimension=2,
        api_key="test-only-key",
        base_url="https://embedding.invalid/v1",
    )

    assert service._client is fake_client
    assert captured == {
        "api_key": "test-only-key",
        "base_url": "https://embedding.invalid/v1",
        "max_retries": 0,
    }


class _ControlledCloseClient:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.fail_once = False

    async def close(self) -> None:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("close failed")


@pytest.mark.asyncio
async def test_embedding_close_is_shared_cancellation_safe_and_retryable() -> None:
    client = _ControlledCloseClient()
    service = OpenAICompatibleRagEmbeddingService(
        provider="compatible",
        model="model",
        version="v1",
        dimension=2,
        client=client,
    )
    first = asyncio.create_task(service.aclose())
    second = asyncio.create_task(service.aclose())
    await client.started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    client.release.set()
    await second
    await service.aclose()
    assert client.calls == 1
    assert service._closed is True

    retry_client = _ControlledCloseClient()
    retry_client.fail_once = True
    retry_client.release.set()
    retry_service = OpenAICompatibleRagEmbeddingService(
        provider="compatible",
        model="model",
        version="v1",
        dimension=2,
        client=retry_client,
    )
    with pytest.raises(RuntimeError, match="close failed"):
        await retry_service.aclose()
    assert retry_service._closed is False
    await retry_service.aclose()
    assert retry_client.calls == 2
    assert retry_service._closed is True
