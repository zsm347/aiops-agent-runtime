from __future__ import annotations

from types import SimpleNamespace

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.persistence.repositories.rag import (
    RagKnowledgeBaseContractError,
    RagPersistenceError,
)
from superbiz_agent.rag.embedding import RagQueryEmbedding
from superbiz_agent.rag.milvus_store import (
    MilvusStoreError,
    MilvusStoreIsolationError,
    RagHybridSearchHit,
)
from superbiz_agent.rag.models import RagRetrievalRequest, RagRetrievalScope
from superbiz_agent.rag.retrieval import (
    MilvusRagRetrievalService,
    RagKnowledgeBaseNotConfiguredError,
    RagRetrievalContractError,
    RagRetrievalIsolationError,
    RagRetrievalUnavailableError,
    RepositoryDefaultKnowledgeBaseResolver,
)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        rag_fixture_mode=False,
        rag_embedding_provider="local-deterministic",
        rag_embedding_model="local-deterministic",
        rag_embedding_version="test-v1",
        rag_embedding_dimension=4,
        rag_hybrid_top_k=3,
        rag_final_top_k=2,
    )


def _request(query: str = "中文订单告警") -> RagRetrievalRequest:
    return RagRetrievalRequest(
        query=query,
        scope=RagRetrievalScope(
            tenant_id="tenant-a",
            user_id="user-a",
            agent_id="agent-a",
            run_id="run-a",
            tool_call_id="call-a",
        ),
    )


def _embedding(**overrides) -> RagQueryEmbedding:
    values = {
        "vector": (1.0, 0.0, 0.0, 0.0),
        "provider": "local-deterministic",
        "model": "local-deterministic",
        "version": "test-v1",
        "dimension": 4,
    }
    values.update(overrides)
    return RagQueryEmbedding(**values)


def _hit(
    chunk_id: str,
    document_id: str,
    *,
    score: float,
    tenant_id: str = "tenant-a",
    knowledge_base_id: str = "kb-a",
) -> RagHybridSearchHit:
    return RagHybridSearchHit(
        chunk_id=chunk_id,
        document_id=document_id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_name=f"{document_id}.md",
        content=f"Evidence for {document_id}",
        score=score,
        heading_path=("Runbook", document_id),
        section_title=document_id,
        page_start=None,
        page_end=2,
    )


class _Resolver:
    def __init__(self, knowledge_base_id: str | None = "kb-a", error: Exception | None = None):
        self.knowledge_base_id = knowledge_base_id
        self.error = error
        self.calls = []

    async def resolve(self, tenant_id: str) -> str | None:
        self.calls.append(tenant_id)
        if self.error is not None:
            raise self.error
        return self.knowledge_base_id


class _EmbeddingService:
    def __init__(self, embedding: RagQueryEmbedding | None = None, error: Exception | None = None):
        self.embedding = embedding or _embedding()
        self.error = error
        self.queries = []

    async def embed_query(self, text: str) -> RagQueryEmbedding:
        self.queries.append(text)
        if self.error is not None:
            raise self.error
        return self.embedding

    async def embed_documents(self, texts):
        raise AssertionError(f"document embedding is not expected: {texts}")


class _ChunkStore:
    def __init__(self, hits=(), error: Exception | None = None):
        self.hits = tuple(hits)
        self.error = error
        self.calls = []

    async def hybrid_search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.hits


class _Documents:
    def __init__(self, statuses=None, error: Exception | None = None):
        self.statuses = statuses or {}
        self.error = error
        self.calls = []

    async def get_document_statuses(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return dict(self.statuses)


def _service(*, resolver=None, embedding=None, store=None, documents=None):
    return MilvusRagRetrievalService(
        settings=_settings(),
        knowledge_base_resolver=resolver or _Resolver(),
        document_repository=documents or _Documents(),
        embedding_service=embedding or _EmbeddingService(),
        chunk_store=store or _ChunkStore(),
    )


@pytest.mark.asyncio
async def test_retrieval_preserves_rrf_order_filters_active_and_builds_citations() -> None:
    hits = (
        _hit("chunk-a", "document-a", score=0.0327),
        _hit("chunk-b", "document-b", score=0.0310),
        _hit("chunk-c", "document-c", score=0.0300),
    )
    embedding = _EmbeddingService()
    store = _ChunkStore(hits)
    documents = _Documents(
        {"document-a": "active", "document-b": "failed", "document-c": "active"}
    )
    service = _service(embedding=embedding, store=store, documents=documents)

    result = await service.search(_request())

    assert [chunk.id for chunk in result.chunks] == ["chunk-a", "chunk-c"]
    assert [chunk.score for chunk in result.chunks] == [0.0327, 0.0300]
    assert result.chunks[0].source == "document-a.md"
    assert result.chunks[0].metadata == {
        "documentId": "document-a",
        "documentName": "document-a.md",
        "knowledgeBaseId": "kb-a",
        "headingPath": ["Runbook", "document-a"],
        "sectionTitle": "document-a",
        "pageStart": None,
        "pageEnd": 2,
        "retrievalSource": "hybrid",
        "confidence": "unavailable",
        "evidenceType": "untrusted_retrieved_document",
    }
    assert store.calls == [
        {
            "query_str": "中文订单告警",
            "query_embedding": _embedding(),
            "tenant_id": "tenant-a",
            "knowledge_base_id": "kb-a",
            "similarity_top_k": 3,
        }
    ]
    assert documents.calls[0]["document_ids"] == (
        "document-a",
        "document-b",
        "document-c",
    )


@pytest.mark.asyncio
async def test_no_default_has_zero_embedding_milvus_and_database_status_side_effects() -> None:
    embedding = _EmbeddingService()
    store = _ChunkStore()
    documents = _Documents()
    service = _service(
        resolver=_Resolver(knowledge_base_id=None),
        embedding=embedding,
        store=store,
        documents=documents,
    )

    with pytest.raises(RagKnowledgeBaseNotConfiguredError):
        await service.search(_request())
    assert embedding.queries == []
    assert store.calls == []
    assert documents.calls == []


@pytest.mark.asyncio
async def test_no_hits_returns_no_results_without_document_status_query() -> None:
    documents = _Documents()
    result = await _service(store=_ChunkStore(), documents=documents).search(_request())

    assert result.chunks == ()
    assert documents.calls == []


@pytest.mark.asyncio
async def test_missing_document_status_is_contract_failure_not_inactive() -> None:
    service = _service(
        store=_ChunkStore([_hit("chunk-a", "document-a", score=0.03)]),
        documents=_Documents({}),
    )
    with pytest.raises(RagRetrievalContractError):
        await service.search(_request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statuses",
    [
        {"document-a": "active", "unexpected": "active"},
        {"document-a": "unknown"},
        {"document-a": ["active"]},
    ],
)
async def test_document_status_contract_rejects_extra_ids_and_unknown_states(statuses) -> None:
    service = _service(
        store=_ChunkStore([_hit("chunk-a", "document-a", score=0.03)]),
        documents=_Documents(statuses),
    )
    with pytest.raises(RagRetrievalContractError):
        await service.search(_request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "embedding",
    [
        _embedding(provider="other"),
        _embedding(model="other"),
        _embedding(version="other"),
        _embedding(dimension=3, vector=(1.0, 0.0, 0.0)),
        _embedding(vector=(True, 0.0, 0.0, 0.0)),
    ],
)
async def test_embedding_identity_and_shape_fail_before_milvus(
    embedding: RagQueryEmbedding,
) -> None:
    store = _ChunkStore()
    service = _service(embedding=_EmbeddingService(embedding), store=store)
    with pytest.raises(RagRetrievalContractError):
        await service.search(_request())
    assert store.calls == []


@pytest.mark.asyncio
async def test_none_embedding_vector_is_a_non_retryable_contract_failure() -> None:
    malformed = RagQueryEmbedding(
        vector=None,  # type: ignore[arg-type]
        provider="local-deterministic",
        model="local-deterministic",
        version="test-v1",
        dimension=4,
    )
    store = _ChunkStore()
    service = _service(embedding=_EmbeddingService(malformed), store=store)

    with pytest.raises(RagRetrievalContractError):
        await service.search(_request())
    assert store.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hits", [None, "not-hits", [SimpleNamespace(document_id="doc")]])
async def test_malformed_adapter_result_is_a_contract_failure(hits) -> None:
    store = _ChunkStore()
    store.hits = hits
    with pytest.raises(RagRetrievalContractError):
        await _service(store=store).search(_request())


@pytest.mark.asyncio
async def test_store_isolation_contract_and_unavailable_errors_are_typed() -> None:
    with pytest.raises(RagRetrievalIsolationError):
        await _service(store=_ChunkStore(error=MilvusStoreIsolationError())).search(_request())
    with pytest.raises(RagRetrievalContractError):
        await _service(store=_ChunkStore(error=MilvusStoreError())).search(_request())
    with pytest.raises(RagRetrievalUnavailableError):
        await _service(store=_ChunkStore(error=ConnectionError())).search(_request())


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", " ", "x" * 2001])
async def test_invalid_query_fails_before_all_dependencies(query: str) -> None:
    resolver = _Resolver()
    service = _service(resolver=resolver)
    with pytest.raises(RagRetrievalContractError):
        await service.search(_request(query))
    assert resolver.calls == []


class _KnowledgeBaseRepository:
    def __init__(self, record=None, error: Exception | None = None):
        self.record = record
        self.error = error

    async def get_default_for_tenant(self, *, tenant_id: str):
        if self.error is not None:
            raise self.error
        return self.record


@pytest.mark.asyncio
async def test_repository_resolver_validates_scope_uniqueness_and_availability() -> None:
    valid = SimpleNamespace(
        id="kb-a", tenant_id="tenant-a", status="active", is_default=True
    )
    assert await RepositoryDefaultKnowledgeBaseResolver(
        _KnowledgeBaseRepository(valid)
    ).resolve("tenant-a") == "kb-a"

    invalid = SimpleNamespace(
        id="kb-b", tenant_id="tenant-b", status="active", is_default=True
    )
    with pytest.raises(RagRetrievalContractError):
        await RepositoryDefaultKnowledgeBaseResolver(
            _KnowledgeBaseRepository(invalid)
        ).resolve("tenant-a")
    with pytest.raises(RagRetrievalContractError):
        await RepositoryDefaultKnowledgeBaseResolver(
            _KnowledgeBaseRepository(error=RagKnowledgeBaseContractError())
        ).resolve("tenant-a")
    with pytest.raises(RagRetrievalUnavailableError):
        await RepositoryDefaultKnowledgeBaseResolver(
            _KnowledgeBaseRepository(error=RagPersistenceError("db", "secret"))
        ).resolve("tenant-a")


@pytest.mark.asyncio
async def test_final_tool_json_over_50000_characters_fails_closed() -> None:
    hits = [
        RagHybridSearchHit(
            chunk_id=f"chunk-{index}",
            document_id=f"document-{index}",
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            document_name=f"document-{index}.md",
            content="x" * 30000,
            score=0.03,
            heading_path=("Runbook",),
            section_title="Runbook",
            page_start=None,
            page_end=None,
        )
        for index in range(2)
    ]
    service = _service(
        store=_ChunkStore(hits),
        documents=_Documents({"document-0": "active", "document-1": "active"}),
    )
    with pytest.raises(RagRetrievalContractError):
        await service.search(_request())


@pytest.mark.asyncio
async def test_forged_top_k_1001_fails_before_resolver_embedding_and_store() -> None:
    settings = _settings().model_copy(update={"rag_hybrid_top_k": 1001})
    resolver = _Resolver()
    embedding = _EmbeddingService()
    store = _ChunkStore()
    service = MilvusRagRetrievalService(
        settings=settings,
        knowledge_base_resolver=resolver,
        document_repository=_Documents(),
        embedding_service=embedding,
        chunk_store=store,
    )

    with pytest.raises(RagRetrievalContractError) as exc_info:
        await service.search(_request())
    assert exc_info.value.retryable is False
    assert exc_info.value.retryable_by_model is False
    assert resolver.calls == []
    assert embedding.queries == []
    assert store.calls == []
