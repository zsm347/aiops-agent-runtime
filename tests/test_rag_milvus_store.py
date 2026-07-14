from __future__ import annotations

import asyncio
import importlib.metadata
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode, VectorStoreQueryResult
from llama_index.vector_stores.milvus.base import _to_milvus_filter
from pymilvus import DataType, FunctionType, MilvusClient

from superbiz_agent.rag.embedding import (
    DeterministicRagEmbeddingService,
    RagQueryEmbedding,
)
from superbiz_agent.rag.milvus_store import (
    BM25_FUNCTION_NAME,
    DENSE_FIELD,
    DOCUMENT_ID_FIELD,
    ID_FIELD,
    JIEBA_ANALYZER_PARAMS,
    OUTPUT_FIELDS,
    RRF_K,
    SCALAR_FIELDS,
    SPARSE_FIELD,
    TEXT_FIELD,
    VARCHAR_MAX_LENGTH,
    MilvusHybridChunkStore,
    MilvusStoreClosedError,
    MilvusStoreError,
    MilvusStoreIsolationError,
    encode_milvus_filter_value,
)
from superbiz_agent.rag.models import RagPreparedChunk


def _chunk(
    chunk_id: str = "chunk-1",
    *,
    tenant_id: str = "tenant-a",
    knowledge_base_id: str = "kb-1",
    document_id: str = "document-1",
    content: str = "中文告警处理",
):
    return RagPreparedChunk(
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        document_name="runbook.md",
        source_uri="kb://runbook",
        heading_path=("告警", "处理"),
        section_title=None,
        page_start=None,
        page_end=2,
        page_metadata={"source": "internal"},
        chunk_index=0,
        content_hash="hash-1",
        content=content,
    )


class _FakeVectorStore:
    def __init__(self, returned_ids):
        self.returned_ids = returned_ids
        self.nodes = None

    async def async_add(self, nodes):
        self.nodes = nodes
        return self.returned_ids


class _PostflightRaceClient:
    def __init__(self, phase: str, events: list[str]):
        self.phase = phase
        self.events = events

    def has_collection(self, collection_name: str) -> bool:
        assert collection_name == "chunks"
        self.events.append(f"{self.phase}.has_collection")
        return self.phase == "postflight"

    def describe_collection(self, collection_name: str):
        assert collection_name == "chunks"
        self.events.append(f"{self.phase}.describe_collection")
        return {"auto_id": False, "fields": [], "functions": []}

    def close(self) -> None:
        self.events.append(f"{self.phase}.close")


@pytest.mark.asyncio
async def test_upsert_maps_textnode_source_scalars_and_uses_async_add() -> None:
    chunk = _chunk()
    batch = await DeterministicRagEmbeddingService(4).embed_documents([chunk.content])
    fake = _FakeVectorStore([chunk.chunk_id])
    store = MilvusHybridChunkStore(
        uri="unused",
        collection_name="chunks",
        dimension=4,
        now=lambda: datetime(2026, 7, 12, 1, 2, 3, tzinfo=timezone.utc),
    )
    store._store = fake

    assert await store.upsert([chunk], batch) == (chunk.chunk_id,)
    node = fake.nodes[0]
    assert node.node_id == chunk.chunk_id
    assert node.text == chunk.content
    assert node.ref_doc_id == chunk.document_id
    assert node.metadata == {
        "tenant_id": "tenant-a",
        "knowledge_base_id": "kb-1",
        "document_name": "runbook.md",
        "source_uri": "kb://runbook",
        "heading_path_json": '["告警","处理"]',
        "section_title": "",
        "page_start": 0,
        "page_end": 2,
        "page_metadata_json": '{"source":"internal"}',
        "chunk_index": 0,
        "content_hash": "hash-1",
        "embedding_provider": "local-deterministic",
        "embedding_model": "local-deterministic",
        "embedding_version": "test-v1",
        "embedding_dimension": 4,
        "created_at": "2026-07-12T01:02:03Z",
    }


@pytest.mark.asyncio
async def test_upsert_rejects_framework_id_mismatch() -> None:
    chunk = _chunk()
    batch = await DeterministicRagEmbeddingService(4).embed_documents([chunk.content])
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._store = _FakeVectorStore(["different"])
    with pytest.raises(MilvusStoreError, match="IDs"):
        await store.upsert([chunk], batch)


@pytest.mark.asyncio
async def test_existing_incompatible_collection_fails_closed(tmp_path: Path) -> None:
    uri = str(tmp_path / "bad.db")
    client = MilvusClient(uri)
    client.create_collection("chunks", dimension=3)
    client.close()
    store = MilvusHybridChunkStore(uri=uri, collection_name="chunks", dimension=4)
    with pytest.raises(MilvusStoreError, match="field"):
        await store.ensure_ready()


@pytest.mark.asyncio
async def test_postflight_rejects_collection_created_after_preflight() -> None:
    events: list[str] = []
    phases = iter(("preflight", "postflight"))

    def client_factory(**kwargs):
        assert kwargs == {"uri": "unused", "token": ""}
        return _PostflightRaceClient(next(phases), events)

    def vector_store_factory(**kwargs):
        assert kwargs["overwrite"] is False
        events.append("vector_store.construct")
        return _FakeVectorStore([])

    store = MilvusHybridChunkStore(
        uri="unused",
        collection_name="chunks",
        dimension=4,
        client_factory=client_factory,
        vector_store_factory=vector_store_factory,
    )

    with pytest.raises(MilvusStoreError, match="field"):
        await store.ensure_ready()

    assert events == [
        "preflight.has_collection",
        "preflight.close",
        "vector_store.construct",
        "postflight.has_collection",
        "postflight.describe_collection",
        "postflight.close",
    ]


def _compatible_collection_description() -> dict:
    fields = []
    required = {
        ID_FIELD: DataType.VARCHAR,
        DOCUMENT_ID_FIELD: DataType.VARCHAR,
        TEXT_FIELD: DataType.VARCHAR,
        DENSE_FIELD: DataType.FLOAT_VECTOR,
        SPARSE_FIELD: DataType.SPARSE_FLOAT_VECTOR,
        **dict(SCALAR_FIELDS),
    }
    for name, field_type in required.items():
        params = {}
        if field_type == DataType.VARCHAR:
            params["max_length"] = VARCHAR_MAX_LENGTH
        if name == TEXT_FIELD:
            params.update(
                {
                    "enable_analyzer": "true",
                    "analyzer_params": json.dumps(JIEBA_ANALYZER_PARAMS),
                }
            )
        if name == DENSE_FIELD:
            params["dim"] = 4
        fields.append(
            {
                "name": name,
                "type": field_type,
                "is_primary": name == ID_FIELD,
                "params": params,
            }
        )
    return {
        "auto_id": False,
        "fields": fields,
        "functions": [
            {
                "name": BM25_FUNCTION_NAME,
                "type": FunctionType.BM25,
                "input_field_names": [TEXT_FIELD],
                "output_field_names": [SPARSE_FIELD],
            }
        ],
    }


class _SchemaClient:
    def __init__(self, description: dict) -> None:
        self.description = description

    def describe_collection(self, collection_name: str) -> dict:
        assert collection_name == "chunks"
        return self.description

    def list_indexes(self, collection_name: str) -> list[str]:
        assert collection_name == "chunks"
        return ["dense-index", "sparse-index"]

    def describe_index(self, collection_name: str, index_name: str) -> dict:
        assert collection_name == "chunks"
        if index_name == "dense-index":
            return {
                "field_name": DENSE_FIELD,
                "index_type": "FLAT",
                "metric_type": "COSINE",
            }
        return {
            "field_name": SPARSE_FIELD,
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
        }


@pytest.mark.parametrize("incompatibility", ["auto_id", "primary_key", "varchar_length"])
def test_collection_preflight_rejects_identity_and_varchar_contract(incompatibility: str) -> None:
    description = _compatible_collection_description()
    if incompatibility == "auto_id":
        description["auto_id"] = True
    elif incompatibility == "primary_key":
        next(field for field in description["fields"] if field["name"] == ID_FIELD)[
            "is_primary"
        ] = False
    else:
        next(field for field in description["fields"] if field["name"] == DOCUMENT_ID_FIELD)[
            "params"
        ]["max_length"] = 128
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)

    with pytest.raises(MilvusStoreError):
        store._validate_collection_safely(_SchemaClient(description))


@pytest.mark.asyncio
async def test_milvus_lite_real_upsert_read_tenant_scalar_and_duplicate_id(tmp_path: Path) -> None:
    uri = str(tmp_path / "rag-lite.db")
    store = MilvusHybridChunkStore(uri=uri, collection_name="chunks", dimension=8)
    embedding = DeterministicRagEmbeddingService(8)
    first_chunks = [
        _chunk("chunk-a", tenant_id="tenant-a", content="中文订单告警处理"),
        _chunk("chunk-b", tenant_id="tenant-b", content="库存告警处理"),
    ]
    first_batch = await embedding.embed_documents([chunk.content for chunk in first_chunks])
    assert await store.upsert(first_chunks, first_batch) == ("chunk-a", "chunk-b")

    client = MilvusClient(uri)
    rows = client.get("chunks", ["chunk-a", "chunk-b"], output_fields=["text", "tenant_id"])
    assert {row["id"] for row in rows} == {"chunk-a", "chunk-b"}
    tenant_rows = client.query(
        "chunks", filter='tenant_id == "tenant-a"', output_fields=["id", "tenant_id"]
    )
    assert tenant_rows == [{"id": "chunk-a", "tenant_id": "tenant-a"}]

    replacement = _chunk("chunk-a", tenant_id="tenant-a", content="中文订单告警已更新")
    replacement_batch = await embedding.embed_documents([replacement.content])
    assert await store.upsert([replacement], replacement_batch) == ("chunk-a",)
    replaced = client.get("chunks", ["chunk-a"], output_fields=["text", "tenant_id"])
    assert replaced[0]["text"] == "中文订单告警已更新"

    description = client.describe_collection("chunks")
    functions = {function["name"]: function for function in description["functions"]}
    assert set(functions) == {BM25_FUNCTION_NAME}
    bm25 = functions[BM25_FUNCTION_NAME]
    assert bm25["type"].name == "BM25"
    assert bm25["input_field_names"] == [TEXT_FIELD]
    assert bm25["output_field_names"] == [SPARSE_FIELD]

    fields = {field["name"]: field for field in description["fields"]}
    text_params = fields[TEXT_FIELD]["params"]
    assert text_params["enable_analyzer"] == "true"
    assert json.loads(text_params["analyzer_params"]) == JIEBA_ANALYZER_PARAMS
    for field_name, field_type in SCALAR_FIELDS:
        assert fields[field_name]["type"] == field_type

    indexes = {
        item["field_name"]: item
        for name in client.list_indexes("chunks")
        for item in [client.describe_index("chunks", name)]
    }
    assert indexes[DENSE_FIELD]["index_type"] == "FLAT"
    assert indexes[DENSE_FIELD]["metric_type"] == "COSINE"
    assert indexes[SPARSE_FIELD]["index_type"] == "SPARSE_INVERTED_INDEX"
    assert indexes[SPARSE_FIELD]["metric_type"] == "BM25"
    client.close()


def _query_embedding(dimension: int = 4) -> RagQueryEmbedding:
    return RagQueryEmbedding(
        vector=tuple(1.0 if index == 0 else 0.0 for index in range(dimension)),
        provider="local-deterministic",
        model="local-deterministic",
        version="test-v1",
        dimension=dimension,
    )


def _hit_node(
    *,
    node_id: str = "random-framework-node-id",
    tenant_id: str = "tenant-a",
    knowledge_base_id: str = "kb-1",
    content: str = "中文告警处理",
    metadata_overrides: dict | None = None,
) -> TextNode:
    metadata = {
        "document_id": "document-1",
        "tenant_id": tenant_id,
        "knowledge_base_id": knowledge_base_id,
        "document_name": "runbook.md",
        "source_uri": "kb://runbook",
        "heading_path_json": '["告警","处理"]',
        "section_title": "处理",
        "page_start": 0,
        "page_end": 2,
        "page_metadata_json": '{"source":"internal"}',
        "chunk_index": 0,
        "content_hash": "hash-1",
        "embedding_provider": "local-deterministic",
        "embedding_model": "local-deterministic",
        "embedding_version": "test-v1",
        "embedding_dimension": 4,
        "created_at": "2026-07-12T01:02:03Z",
    }
    metadata.update(metadata_overrides or {})
    return TextNode(id_=node_id, text=content, metadata=metadata)


class _FakeSearchStore:
    def __init__(self, result: VectorStoreQueryResult) -> None:
        self.result = result
        self.queries = []

    async def aquery(self, query):
        self.queries.append(query)
        return self.result


@pytest.mark.asyncio
async def test_hybrid_search_builds_structured_query_and_uses_result_ids() -> None:
    fake = _FakeSearchStore(
        VectorStoreQueryResult(
            ids=["stable-chunk-id"],
            nodes=[
                _hit_node(
                    tenant_id="tenant\\scope",
                    knowledge_base_id="kb\\scope",
                )
            ],
            similarities=[2 / 61],
        )
    )
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._store = fake

    hits = await store.hybrid_search(
        query_str="中文告警",
        query_embedding=_query_embedding(),
        tenant_id="tenant\\scope",
        knowledge_base_id="kb\\scope",
        similarity_top_k=10,
    )

    query = fake.queries[0]
    assert query.mode is VectorStoreQueryMode.HYBRID
    assert query.query_embedding == [1.0, 0.0, 0.0, 0.0]
    assert query.query_str == "中文告警"
    assert query.similarity_top_k == 10
    assert query.filters.condition is FilterCondition.AND
    assert [(item.key, item.value, item.operator) for item in query.filters.filters] == [
        ("tenant_id", "tenant\\\\scope", FilterOperator.EQ),
        ("knowledge_base_id", "kb\\\\scope", FilterOperator.EQ),
    ]
    assert hits[0].chunk_id == "stable-chunk-id"
    assert hits[0].chunk_id != fake.result.nodes[0].node_id
    assert hits[0].score == pytest.approx(2 / 61)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_str", ""),
        ("tenant_id", ""),
        ("knowledge_base_id", " "),
        ("similarity_top_k", 0),
        ("similarity_top_k", True),
        ("similarity_top_k", 1001),
    ],
)
async def test_hybrid_search_rejects_bad_input_before_rpc(field: str, value) -> None:
    fake = _FakeSearchStore(VectorStoreQueryResult(ids=[], nodes=[], similarities=[]))
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._store = fake
    kwargs = {
        "query_str": "query",
        "query_embedding": _query_embedding(),
        "tenant_id": "tenant-a",
        "knowledge_base_id": "kb-1",
        "similarity_top_k": 3,
    }
    kwargs[field] = value

    with pytest.raises(MilvusStoreError):
        await store.hybrid_search(**kwargs)
    assert fake.queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "embedding",
    [
        RagQueryEmbedding((1.0,), "local-deterministic", "local-deterministic", "test-v1", 1),
        RagQueryEmbedding(
            (True, 0.0, 0.0, 0.0),
            "local-deterministic",
            "local-deterministic",
            "test-v1",
            4,
        ),
        RagQueryEmbedding(
            (float("nan"), 0.0, 0.0, 0.0),
            "local-deterministic",
            "local-deterministic",
            "test-v1",
            4,
        ),
    ],
)
async def test_hybrid_search_rejects_bad_query_embedding_before_rpc(
    embedding: RagQueryEmbedding,
) -> None:
    fake = _FakeSearchStore(VectorStoreQueryResult(ids=[], nodes=[], similarities=[]))
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._store = fake
    with pytest.raises(MilvusStoreError):
        await store.hybrid_search(
            query_str="query",
            query_embedding=embedding,
            tenant_id="tenant-a",
            knowledge_base_id="kb-1",
            similarity_top_k=3,
        )
    assert fake.queries == []


@pytest.mark.asyncio
async def test_hybrid_search_rejects_none_query_vector_before_rpc() -> None:
    fake = _FakeSearchStore(VectorStoreQueryResult(ids=[], nodes=[], similarities=[]))
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._store = fake
    malformed = RagQueryEmbedding(
        vector=None,  # type: ignore[arg-type]
        provider="local-deterministic",
        model="local-deterministic",
        version="test-v1",
        dimension=4,
    )
    with pytest.raises(MilvusStoreError, match="vector"):
        await store.hybrid_search(
            query_str="query",
            query_embedding=malformed,
            tenant_id="tenant-a",
            knowledge_base_id="kb-1",
            similarity_top_k=3,
        )
    assert fake.queries == []


@pytest.mark.asyncio
async def test_hybrid_search_fails_closed_on_scope_mismatch() -> None:
    fake = _FakeSearchStore(
        VectorStoreQueryResult(
            ids=["chunk-1"],
            nodes=[_hit_node(tenant_id="tenant-b")],
            similarities=[0.03],
        )
    )
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._store = fake

    with pytest.raises(MilvusStoreIsolationError):
        await store.hybrid_search(
            query_str="query",
            query_embedding=_query_embedding(),
            tenant_id="tenant-a",
            knowledge_base_id="kb-1",
            similarity_top_k=3,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        None,
        VectorStoreQueryResult(ids=None, nodes=[], similarities=[]),
        VectorStoreQueryResult(ids=["a"], nodes=[], similarities=[0.1]),
        VectorStoreQueryResult(ids=[["unhashable"]], nodes=[_hit_node()], similarities=[0.1]),
        VectorStoreQueryResult(
            ids=["a", "a"], nodes=[_hit_node(), _hit_node()], similarities=[0.1, 0.2]
        ),
        VectorStoreQueryResult(ids=["a"], nodes=[_hit_node()], similarities=[float("nan")]),
        VectorStoreQueryResult(ids=[1], nodes=[_hit_node()], similarities=[0.1]),
        VectorStoreQueryResult(ids=["a"], nodes=[SimpleNamespace(metadata=None)], similarities=[0.1]),
        VectorStoreQueryResult(
            ids=["a"],
            nodes=[_hit_node(metadata_overrides={"embedding_version": "other"})],
            similarities=[0.1],
        ),
        VectorStoreQueryResult(
            ids=["a"],
            nodes=[_hit_node(metadata_overrides={"heading_path_json": "not-json"})],
            similarities=[0.1],
        ),
    ],
)
async def test_hybrid_search_rejects_invalid_result_contract(result) -> None:
    fake = _FakeSearchStore(result)
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._store = fake
    with pytest.raises(MilvusStoreError):
        await store.hybrid_search(
            query_str="query",
            query_embedding=_query_embedding(),
            tenant_id="tenant-a",
            knowledge_base_id="kb-1",
            similarity_top_k=3,
        )


def test_d1_dependency_versions_are_exact() -> None:
    expected = {
        "llama-index-core": "0.14.23",
        "llama-index-vector-stores-milvus": "1.1.0",
        "pymilvus": "2.6.16",
        "milvus-lite": "3.0",
        "jieba": "0.42.1",
        "openai": "2.45.0",
    }
    assert {name: importlib.metadata.version(name) for name in expected} == expected


class _ClosableClient:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _AsyncClosableClient:
    def __init__(self) -> None:
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


class _InitializedStore:
    def __init__(self) -> None:
        self._milvusclient = _ClosableClient()
        self._async_milvusclient = _AsyncClosableClient()


class _RetryableCloseClient:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1
        failure = self.failure
        self.failure = None
        if failure is not None:
            raise failure


class _RetryableSyncCloseClient:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        failure = self.failure
        self.failure = None
        if failure is not None:
            raise failure


class _BlockingCloseClient:
    def __init__(self) -> None:
        self.closed = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def close(self) -> None:
        self.closed += 1
        self.started.set()
        await self.release.wait()


@pytest.mark.asyncio
async def test_shared_initialization_survives_cancelled_waiter() -> None:
    started = threading.Event()
    release = threading.Event()
    initialized = _InitializedStore()
    calls = 0

    def initialize():
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(5)
        return initialized

    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._initialize = initialize
    waiter = asyncio.create_task(store.ensure_ready())
    assert await asyncio.to_thread(started.wait, 5)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()

    await store.ensure_ready()
    assert calls == 1
    assert store._store is initialized
    await store.aclose()


@pytest.mark.asyncio
async def test_shared_initialization_runs_once_for_concurrent_waiters() -> None:
    initialized = _InitializedStore()
    calls = 0

    def initialize():
        nonlocal calls
        calls += 1
        return initialized

    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._initialize = initialize
    await asyncio.gather(*(store.ensure_ready() for _ in range(10)))

    assert calls == 1
    await store.aclose()


@pytest.mark.asyncio
async def test_transient_init_failure_can_retry_but_contract_failure_is_cached() -> None:
    transient_store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    initialized = _InitializedStore()
    transient_calls = 0

    def transient_initialize():
        nonlocal transient_calls
        transient_calls += 1
        if transient_calls == 1:
            raise ConnectionError("temporary")
        return initialized

    transient_store._initialize = transient_initialize
    with pytest.raises(ConnectionError):
        await transient_store.ensure_ready()
    await transient_store.ensure_ready()
    assert transient_calls == 2
    await transient_store.aclose()

    contract_store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    contract_calls = 0

    def contract_initialize():
        nonlocal contract_calls
        contract_calls += 1
        raise MilvusStoreError("schema contract")

    contract_store._initialize = contract_initialize
    for _ in range(2):
        with pytest.raises(MilvusStoreError, match="schema contract"):
            await contract_store.ensure_ready()
    assert contract_calls == 1
    await contract_store.aclose()


@pytest.mark.asyncio
async def test_aclose_waits_for_running_initialization_and_closes_created_clients() -> None:
    started = threading.Event()
    release = threading.Event()
    initialized = _InitializedStore()

    def initialize():
        started.set()
        assert release.wait(5)
        return initialized

    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._initialize = initialize
    waiter = asyncio.create_task(store.ensure_ready())
    assert await asyncio.to_thread(started.wait, 5)
    close_task = asyncio.create_task(store.aclose())
    await asyncio.sleep(0)
    release.set()
    await close_task

    with pytest.raises(MilvusStoreClosedError):
        await waiter
    assert initialized._milvusclient.closed == 1
    assert initialized._async_milvusclient.closed == 1
    with pytest.raises(MilvusStoreClosedError):
        await store.ensure_ready()


@pytest.mark.asyncio
async def test_milvus_close_attempts_all_clients_and_retries_partial_failure() -> None:
    initialized = _InitializedStore()
    initialized._async_milvusclient = _RetryableCloseClient(asyncio.CancelledError())
    initialized._milvusclient = _RetryableSyncCloseClient(RuntimeError("sync failed"))
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._store = initialized

    with pytest.raises(MilvusStoreError, match="async_client, sync_client"):
        await store.aclose()
    assert store._closed is False
    assert initialized._async_milvusclient.closed == 1
    assert initialized._milvusclient.closed == 1

    await asyncio.gather(store.aclose(), store.aclose())
    await store.aclose()
    assert store._closed is True
    assert initialized._async_milvusclient.closed == 2
    assert initialized._milvusclient.closed == 2


@pytest.mark.asyncio
async def test_milvus_close_continues_after_waiter_cancellation() -> None:
    initialized = _InitializedStore()
    blocking = _BlockingCloseClient()
    initialized._async_milvusclient = blocking
    store = MilvusHybridChunkStore(uri="unused", collection_name="chunks", dimension=4)
    store._store = initialized

    waiter = asyncio.create_task(store.aclose())
    await blocking.started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    blocking.release.set()
    await store.aclose()

    assert store._closed is True
    assert blocking.closed == 1
    assert initialized._milvusclient.closed == 1


class _CapturingHybridClient:
    def __init__(self, entity: dict) -> None:
        self.entity = entity
        self.requests = None
        self.ranker = None
        self.output_fields = None

    async def hybrid_search(
        self,
        collection_name,
        requests,
        *,
        ranker,
        limit,
        output_fields,
        partition_names,
    ):
        assert collection_name == "chunks"
        assert limit == 3
        assert partition_names is None
        self.requests = requests
        self.ranker = ranker
        self.output_fields = output_fields
        return [[{"id": "stable-id", "distance": 2 / 61, "entity": self.entity}]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tenant_id", "knowledge_base_id", "expected_expr"),
    [
        (
            "tenant-normal",
            "kb-normal",
            "(tenant_id == 'tenant-normal' and knowledge_base_id == 'kb-normal')",
        ),
        (
            "tenant'quote",
            "kb'quote",
            "(tenant_id == 'tenant\\'quote' and knowledge_base_id == 'kb\\'quote')",
        ),
        (
            "tenant\\path",
            "kb\\path",
            "(tenant_id == 'tenant\\\\path' and knowledge_base_id == 'kb\\\\path')",
        ),
        (
            "tenant\\' or true or tenant_id != 'victim",
            "kb\\' or true",
            "(tenant_id == 'tenant\\\\\\' or true or tenant_id != \\'victim' and "
            "knowledge_base_id == 'kb\\\\\\' or true')",
        ),
        (
            "租户甲",
            "知识库甲",
            "(tenant_id == '租户甲' and knowledge_base_id == '知识库甲')",
        ),
    ],
)
async def test_actual_llamaindex_hybrid_requests_share_frozen_escaped_expression(
    tmp_path: Path,
    tenant_id: str,
    knowledge_base_id: str,
    expected_expr: str,
) -> None:
    store = MilvusHybridChunkStore(
        uri=str(tmp_path / "capture.db"), collection_name="chunks", dimension=4
    )
    await store.ensure_ready()
    assert store._store.output_fields == list(OUTPUT_FIELDS)
    assert store._store.hybrid_ranker == "RRFRanker"
    assert store._store.hybrid_ranker_params == {"k": RRF_K}

    entity = dict(
        _hit_node(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id).metadata
    )
    entity["text"] = "中文告警处理"
    capture = _CapturingHybridClient(entity)
    original_async_client = store._store._async_milvusclient
    store._store._async_milvusclient = capture
    try:
        hits = await store.hybrid_search(
            query_str="中文告警",
            query_embedding=_query_embedding(),
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            similarity_top_k=3,
        )
    finally:
        store._store._async_milvusclient = original_async_client

    assert [hit.chunk_id for hit in hits] == ["stable-id"]
    dense_request, sparse_request = capture.requests
    assert dense_request.expr.encode("utf-8") == sparse_request.expr.encode("utf-8")
    assert dense_request.expr.encode("utf-8") == expected_expr.encode("utf-8")
    assert dense_request.anns_field == DENSE_FIELD
    assert sparse_request.anns_field == SPARSE_FIELD
    assert capture.ranker.dict() == {"strategy": "rrf", "params": {"k": 60}}
    assert capture.output_fields == [*OUTPUT_FIELDS, TEXT_FIELD]
    await store.aclose()


def _structured_scope_expression(tenant_id: str, knowledge_base_id: str) -> str:
    return _to_milvus_filter(
        MetadataFilters(
            filters=[
                MetadataFilter(
                    key="tenant_id",
                    value=encode_milvus_filter_value(tenant_id),
                    operator=FilterOperator.EQ,
                ),
                MetadataFilter(
                    key="knowledge_base_id",
                    value=encode_milvus_filter_value(knowledge_base_id),
                    operator=FilterOperator.EQ,
                ),
            ],
            condition=FilterCondition.AND,
        )
    )


@pytest.mark.asyncio
async def test_milvus_lite_hybrid_transport_encoding_isolation_and_reopen(
    tmp_path: Path,
) -> None:
    uri = str(tmp_path / "hybrid-scope.db")
    identities = [
        ("tenant-normal", "kb-normal"),
        ("tenant'quote", "kb'quote"),
        ("tenant\\path", "kb\\path"),
        ("tenant\\' or true or tenant_id != 'victim", "kb\\' or true"),
        ("租户甲", "知识库甲"),
    ]
    chunks = [
        _chunk(
            f"chunk-{index}",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_id=f"document-{index}",
            content=f"中文混合检索标记 {index} 订单告警处理",
        )
        for index, (tenant_id, knowledge_base_id) in enumerate(identities)
    ]
    embedding = DeterministicRagEmbeddingService(8)
    store = MilvusHybridChunkStore(uri=uri, collection_name="chunks", dimension=8)
    batch = await embedding.embed_documents([chunk.content for chunk in chunks])
    assert await store.upsert(chunks, batch) == tuple(chunk.chunk_id for chunk in chunks)

    client = MilvusClient(uri)
    for index, ((tenant_id, knowledge_base_id), chunk) in enumerate(zip(identities, chunks)):
        expression = _structured_scope_expression(tenant_id, knowledge_base_id)
        rows = client.query(
            "chunks",
            filter=expression,
            output_fields=["id", "tenant_id", "knowledge_base_id"],
        )
        assert rows == [
            {
                "id": f"chunk-{index}",
                "tenant_id": tenant_id,
                "knowledge_base_id": knowledge_base_id,
            }
        ]

        query_embedding = await embedding.embed_query(chunk.content)
        hits = await store.hybrid_search(
            query_str=chunk.content,
            query_embedding=query_embedding,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            similarity_top_k=10,
        )
        assert [hit.chunk_id for hit in hits] == [f"chunk-{index}"]
        assert all(hit.tenant_id == tenant_id for hit in hits)
        assert all(hit.knowledge_base_id == knowledge_base_id for hit in hits)

    no_results = await store.hybrid_search(
        query_str="不存在的检索内容",
        query_embedding=await embedding.embed_query("不存在的检索内容"),
        tenant_id="missing-tenant",
        knowledge_base_id="missing-kb",
        similarity_top_k=10,
    )
    assert no_results == ()

    replacement = _chunk(
        "chunk-0",
        tenant_id="tenant-normal",
        knowledge_base_id="kb-normal",
        document_id="document-0",
        content="中文订单告警处理内容已经更新",
    )
    replacement_batch = await embedding.embed_documents([replacement.content])
    await store.upsert([replacement], replacement_batch)
    replacement_hits = await store.hybrid_search(
        query_str=replacement.content,
        query_embedding=await embedding.embed_query(replacement.content),
        tenant_id="tenant-normal",
        knowledge_base_id="kb-normal",
        similarity_top_k=10,
    )
    assert [hit.content for hit in replacement_hits] == [replacement.content]
    client.close()
    await store.aclose()


def test_milvus_lite_new_process_reopens_and_loads_existing_collection(tmp_path: Path) -> None:
    uri = str(tmp_path / "cold-reopen.db")
    writer_script = f"""
import asyncio
import sys
sys.path.insert(0, 'src')
from superbiz_agent.rag.embedding import DeterministicRagEmbeddingService
from superbiz_agent.rag.milvus_store import MilvusHybridChunkStore
from superbiz_agent.rag.models import RagPreparedChunk

async def main():
    embedding = DeterministicRagEmbeddingService(8)
    store = MilvusHybridChunkStore(uri={uri!r}, collection_name='chunks', dimension=8)
    text = '冷启动中文订单告警处理'
    chunk = RagPreparedChunk(
        chunk_id='cold-chunk', tenant_id='cold-tenant', knowledge_base_id='cold-kb',
        document_id='cold-document', document_name='cold.md', source_uri='kb://cold',
        heading_path=('冷启动',), section_title='冷启动', page_start=None, page_end=None,
        page_metadata={{}}, chunk_index=0, content_hash='cold-hash', content=text,
    )
    batch = await embedding.embed_documents([text])
    assert await store.upsert([chunk], batch) == ('cold-chunk',)
    await store.aclose()

asyncio.run(main())
"""
    reader_script = f"""
import asyncio
import sys
sys.path.insert(0, 'src')
from superbiz_agent.rag.embedding import DeterministicRagEmbeddingService
from superbiz_agent.rag.milvus_store import MilvusHybridChunkStore

async def main():
    embedding = DeterministicRagEmbeddingService(8)
    store = MilvusHybridChunkStore(uri={uri!r}, collection_name='chunks', dimension=8)
    text = '冷启动中文订单告警处理'
    hits = await store.hybrid_search(
        query_str=text,
        query_embedding=await embedding.embed_query(text),
        tenant_id='cold-tenant',
        knowledge_base_id='cold-kb',
        similarity_top_k=10,
    )
    assert [hit.chunk_id for hit in hits] == ['cold-chunk']
    assert [hit.content for hit in hits] == [text]
    await store.aclose()

asyncio.run(main())
"""
    for script in (writer_script, reader_script):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
