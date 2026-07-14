from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Protocol

from llama_index.core.schema import MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.base import IndexManagement
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from pymilvus import DataType, FunctionType, MilvusClient
from pymilvus.client.types import LoadState

from superbiz_agent.config import RAG_TOP_K_MAX
from superbiz_agent.rag.embedding import RagEmbeddingBatch, RagQueryEmbedding
from superbiz_agent.rag.models import RagPreparedChunk


ID_FIELD = "id"
DOCUMENT_ID_FIELD = "document_id"
TEXT_FIELD = "text"
DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
BM25_FUNCTION_NAME = "rag_text_bm25"
RRF_K = 60
VARCHAR_MAX_LENGTH = 65535

JIEBA_ANALYZER_PARAMS = {
    "tokenizer": {
        "type": "jieba",
        "dict": ["_default_"],
        "mode": "search",
        "hmm": True,
    },
    "filter": ["lowercase", "removepunct"],
}

SCALAR_FIELDS: tuple[tuple[str, DataType], ...] = (
    ("tenant_id", DataType.VARCHAR),
    ("knowledge_base_id", DataType.VARCHAR),
    ("document_name", DataType.VARCHAR),
    ("source_uri", DataType.VARCHAR),
    ("heading_path_json", DataType.VARCHAR),
    ("section_title", DataType.VARCHAR),
    ("page_start", DataType.INT64),
    ("page_end", DataType.INT64),
    ("page_metadata_json", DataType.VARCHAR),
    ("chunk_index", DataType.INT64),
    ("content_hash", DataType.VARCHAR),
    ("embedding_provider", DataType.VARCHAR),
    ("embedding_model", DataType.VARCHAR),
    ("embedding_version", DataType.VARCHAR),
    ("embedding_dimension", DataType.INT64),
    ("created_at", DataType.VARCHAR),
)
OUTPUT_FIELDS: tuple[str, ...] = (
    DOCUMENT_ID_FIELD,
    *(name for name, _ in SCALAR_FIELDS),
)


class MilvusStoreError(RuntimeError):
    """Raised when the Milvus collection or result violates the frozen contract."""


class MilvusStoreIsolationError(PermissionError):
    """Raised when a Milvus hit is outside the requested backend scope."""


class MilvusStoreClosedError(MilvusStoreError):
    """Raised when work is attempted during or after shutdown."""


@dataclass(frozen=True)
class RagHybridSearchHit:
    chunk_id: str
    document_id: str
    tenant_id: str
    knowledge_base_id: str
    document_name: str
    content: str
    score: float
    heading_path: tuple[str, ...]
    section_title: str | None
    page_start: int | None
    page_end: int | None


class RagChunkStore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def upsert(
        self, chunks: Sequence[RagPreparedChunk], embedding_batch: RagEmbeddingBatch
    ) -> tuple[str, ...]: ...


def encode_milvus_filter_value(value: str) -> str:
    """Encode backslashes only for transport through LlamaIndex's Milvus filter adapter."""
    return value.replace("\\", "\\\\")


class MilvusHybridChunkStore:
    def __init__(
        self,
        *,
        uri: str,
        collection_name: str,
        dimension: int,
        token: str | None = None,
        client_factory: Callable[..., Any] = MilvusClient,
        vector_store_factory: Callable[..., Any] = MilvusVectorStore,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        self.uri = uri
        self.token = token or ""
        self.collection_name = collection_name
        self.dimension = dimension
        self._client_factory = client_factory
        self._vector_store_factory = vector_store_factory
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._store: Any | None = None
        self._init_task: asyncio.Task[Any] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._permanent_init_error: MilvusStoreError | None = None
        self._state_lock = asyncio.Lock()
        self._active_operations = 0
        self._operations_done = asyncio.Event()
        self._operations_done.set()
        self._closing = False
        self._closed = False

    async def ensure_ready(self) -> None:
        async with self._state_lock:
            self._raise_if_closing()
            if self._store is not None:
                return
            if self._permanent_init_error is not None:
                raise self._permanent_init_error
            task = self._init_task
            if task is None:
                task = asyncio.create_task(asyncio.to_thread(self._initialize))
                self._init_task = task

        try:
            initialized_store = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done() and task.cancelled():
                async with self._state_lock:
                    if self._init_task is task:
                        self._init_task = None
            raise
        except MilvusStoreError as exc:
            async with self._state_lock:
                if self._init_task is task:
                    self._permanent_init_error = exc
                    self._init_task = None
            raise
        except Exception:
            async with self._state_lock:
                if self._init_task is task:
                    self._init_task = None
            raise

        async with self._state_lock:
            if self._init_task is task:
                self._init_task = None
            if self._closing or self._closed:
                raise MilvusStoreClosedError("Milvus chunk store is closed.")
            if self._store is None:
                self._store = initialized_store

    async def upsert(
        self, chunks: Sequence[RagPreparedChunk], embedding_batch: RagEmbeddingBatch
    ) -> tuple[str, ...]:
        chunk_values = tuple(chunks)
        if not chunk_values:
            raise MilvusStoreError("Milvus upsert requires at least one chunk.")
        if len(chunk_values) != len(embedding_batch.vectors):
            raise MilvusStoreError("Chunk and embedding counts do not match.")
        if embedding_batch.dimension != self.dimension:
            raise MilvusStoreError("Embedding batch dimension does not match the collection.")
        expected_ids = tuple(chunk.chunk_id for chunk in chunk_values)
        if len(set(expected_ids)) != len(expected_ids):
            raise MilvusStoreError("Chunk IDs must be unique within an upsert.")

        await self.ensure_ready()
        store = await self._begin_operation()
        try:
            created_at = self._utc_timestamp()
            nodes = [
                self._to_node(chunk, vector, embedding_batch, created_at)
                for chunk, vector in zip(chunk_values, embedding_batch.vectors, strict=True)
            ]
            returned_ids = tuple(await store.async_add(nodes))
        finally:
            await self._end_operation()
        if len(returned_ids) != len(expected_ids) or set(returned_ids) != set(expected_ids):
            raise MilvusStoreError("Milvus returned IDs that do not match the input chunks.")
        return expected_ids

    async def hybrid_search(
        self,
        *,
        query_str: str,
        query_embedding: RagQueryEmbedding,
        tenant_id: str,
        knowledge_base_id: str,
        similarity_top_k: int,
    ) -> tuple[RagHybridSearchHit, ...]:
        self._validate_search_input(
            query_str=query_str,
            query_embedding=query_embedding,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            similarity_top_k=similarity_top_k,
        )
        query = VectorStoreQuery(
            query_embedding=list(query_embedding.vector),
            query_str=query_str,
            similarity_top_k=similarity_top_k,
            mode=VectorStoreQueryMode.HYBRID,
            filters=MetadataFilters(
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
            ),
        )

        await self.ensure_ready()
        store = await self._begin_operation()
        try:
            result = await store.aquery(query)
        finally:
            await self._end_operation()
        try:
            return self._normalize_search_result(
                result,
                query_embedding=query_embedding,
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            )
        except (MilvusStoreError, MilvusStoreIsolationError):
            raise
        except Exception:
            raise MilvusStoreError("Milvus returned an invalid query result.") from None

    async def aclose(self) -> None:
        async with self._state_lock:
            if self._closed:
                return
            self._closing = True
            task = self._close_task
            if task is None or task.done():
                task = asyncio.create_task(self._close_resources())
                self._close_task = task
        await asyncio.shield(task)

    async def _close_resources(self) -> None:
        async with self._state_lock:
            init_task = self._init_task
            initialized_store = self._store

        if init_task is not None:
            try:
                task_store = await asyncio.shield(init_task)
            except (Exception, asyncio.CancelledError):
                task_store = None
            if initialized_store is None:
                initialized_store = task_store

        if initialized_store is not None:
            async with self._state_lock:
                if self._store is None:
                    self._store = initialized_store

        await self._operations_done.wait()
        if initialized_store is not None:
            await self._close_store(initialized_store)

        async with self._state_lock:
            self._store = None
            self._init_task = None
            self._closed = True

    async def _begin_operation(self) -> Any:
        async with self._state_lock:
            self._raise_if_closing()
            if self._store is None:
                raise MilvusStoreError("Milvus chunk store is not initialized.")
            self._active_operations += 1
            self._operations_done.clear()
            return self._store

    async def _end_operation(self) -> None:
        async with self._state_lock:
            self._active_operations -= 1
            if self._active_operations == 0:
                self._operations_done.set()

    def _raise_if_closing(self) -> None:
        if self._closing or self._closed:
            raise MilvusStoreClosedError("Milvus chunk store is closed.")

    def _initialize(self) -> Any:
        if importlib.util.find_spec("jieba") is None:
            raise MilvusStoreError("The rag extra requires jieba for the fixed BM25 analyzer.")

        client = self._client_factory(uri=self.uri, token=self.token)
        try:
            exists = client.has_collection(self.collection_name)
            if exists:
                self._validate_collection_safely(client)
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                close()

        function = BM25BuiltInFunction(
            function_name=BM25_FUNCTION_NAME,
            input_field_names=TEXT_FIELD,
            output_field_names=SPARSE_FIELD,
            analyzer_params=JIEBA_ANALYZER_PARAMS,
        )
        store = self._vector_store_factory(
            uri=self.uri,
            token=self.token,
            collection_name=self.collection_name,
            overwrite=False,
            upsert_mode=True,
            doc_id_field=DOCUMENT_ID_FIELD,
            text_key=TEXT_FIELD,
            embedding_field=DENSE_FIELD,
            sparse_embedding_field=SPARSE_FIELD,
            enable_dense=True,
            enable_sparse=True,
            sparse_embedding_function=function,
            dim=self.dimension,
            similarity_metric="COSINE",
            index_config={"index_type": "FLAT", "metric_type": "COSINE"},
            sparse_index_config={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
            },
            scalar_field_names=[name for name, _ in SCALAR_FIELDS],
            scalar_field_types=[field_type for _, field_type in SCALAR_FIELDS],
            output_fields=list(OUTPUT_FIELDS),
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": RRF_K},
            index_management=(
                IndexManagement.NO_VALIDATION if exists else IndexManagement.CREATE_IF_NOT_EXISTS
            ),
            use_async_client=True,
        )

        try:
            postflight = self._client_factory(uri=self.uri, token=self.token)
            try:
                if not postflight.has_collection(self.collection_name):
                    raise MilvusStoreError("Milvus collection was not created.")
                self._validate_collection_safely(postflight)
                postflight.load_collection(self.collection_name)
                state = postflight.get_load_state(self.collection_name)
                if state.get("state") != LoadState.Loaded:
                    raise MilvusStoreError("Milvus collection did not reach Loaded state.")
            finally:
                close = getattr(postflight, "close", None)
                if close is not None:
                    close()
        except Exception:
            self._close_store_from_init_thread(store)
            raise
        return store

    def _validate_collection_safely(self, client: Any) -> None:
        try:
            self._validate_collection(client)
        except MilvusStoreError:
            raise
        except Exception:
            raise MilvusStoreError("Milvus collection schema is malformed.") from None

    def _validate_collection(self, client: Any) -> None:
        description = client.describe_collection(self.collection_name)
        if not isinstance(description, Mapping) or description.get("auto_id") is not False:
            raise MilvusStoreError("Milvus collection must use auto_id=false.")
        raw_fields = description.get("fields")
        if isinstance(raw_fields, (str, bytes)) or not isinstance(raw_fields, Sequence):
            raise MilvusStoreError("Milvus collection fields are invalid.")
        if any(not isinstance(field, Mapping) or not isinstance(field.get("name"), str) for field in raw_fields):
            raise MilvusStoreError("Milvus collection fields are invalid.")
        fields = {field["name"]: field for field in raw_fields}
        required_fields = {
            ID_FIELD: DataType.VARCHAR,
            DOCUMENT_ID_FIELD: DataType.VARCHAR,
            TEXT_FIELD: DataType.VARCHAR,
            DENSE_FIELD: DataType.FLOAT_VECTOR,
            SPARSE_FIELD: DataType.SPARSE_FLOAT_VECTOR,
            **dict(SCALAR_FIELDS),
        }
        for name, expected_type in required_fields.items():
            field = fields.get(name)
            if field is None or field.get("type") != expected_type:
                raise MilvusStoreError(f"Milvus field {name!r} is missing or incompatible.")
            if expected_type == DataType.VARCHAR:
                max_length = field.get("params", {}).get("max_length")
                if int(max_length or 0) != VARCHAR_MAX_LENGTH:
                    raise MilvusStoreError(f"Milvus field {name!r} max_length is incompatible.")
        primary_fields = [name for name, field in fields.items() if field.get("is_primary")]
        if primary_fields != [ID_FIELD]:
            raise MilvusStoreError("Milvus id field must be the primary key.")
        dense_dim = fields[DENSE_FIELD].get("params", {}).get("dim")
        if int(dense_dim or 0) != self.dimension:
            raise MilvusStoreError("Milvus dense vector dimension is incompatible.")

        text_params = fields[TEXT_FIELD].get("params", {})
        if str(text_params.get("enable_analyzer", "")).lower() != "true":
            raise MilvusStoreError("Milvus text analyzer is not enabled.")
        analyzer = text_params.get("analyzer_params")
        if isinstance(analyzer, str):
            try:
                analyzer = json.loads(analyzer)
            except json.JSONDecodeError as exc:
                raise MilvusStoreError("Milvus text analyzer configuration is invalid.") from exc
        if analyzer != JIEBA_ANALYZER_PARAMS:
            raise MilvusStoreError("Milvus text analyzer configuration is incompatible.")

        raw_functions = description.get("functions")
        if isinstance(raw_functions, (str, bytes)) or not isinstance(raw_functions, Sequence):
            raise MilvusStoreError("Milvus function schema is invalid.")
        functions = [
            function
            for function in raw_functions
            if isinstance(function, Mapping) and function.get("name") == BM25_FUNCTION_NAME
        ]
        if len(functions) != 1:
            raise MilvusStoreError("Milvus BM25 function is missing or duplicated.")
        function = functions[0]
        if (
            function.get("type") != FunctionType.BM25
            or function.get("input_field_names") != [TEXT_FIELD]
            or function.get("output_field_names") != [SPARSE_FIELD]
        ):
            raise MilvusStoreError("Milvus BM25 function is incompatible.")

        vector_indexes: dict[str, list[dict[str, Any]]] = {DENSE_FIELD: [], SPARSE_FIELD: []}
        for index_name in client.list_indexes(self.collection_name):
            index = client.describe_index(self.collection_name, index_name)
            field_name = index.get("field_name")
            if field_name in vector_indexes:
                vector_indexes[field_name].append(index)
        self._validate_index(vector_indexes[DENSE_FIELD], "FLAT", "COSINE", DENSE_FIELD)
        self._validate_index(
            vector_indexes[SPARSE_FIELD], "SPARSE_INVERTED_INDEX", "BM25", SPARSE_FIELD
        )

    @staticmethod
    def _validate_index(
        indexes: Sequence[dict[str, Any]], index_type: str, metric_type: str, field_name: str
    ) -> None:
        if len(indexes) != 1:
            raise MilvusStoreError(f"Milvus field {field_name!r} must have exactly one index.")
        index = indexes[0]
        if index.get("index_type") != index_type or index.get("metric_type") != metric_type:
            raise MilvusStoreError(f"Milvus index for {field_name!r} is incompatible.")

    def _to_node(
        self,
        chunk: RagPreparedChunk,
        vector: Sequence[float],
        batch: RagEmbeddingBatch,
        created_at: str,
    ) -> TextNode:
        if len(vector) != self.dimension:
            raise MilvusStoreError("Embedding vector dimension does not match the collection.")
        metadata = {
            "tenant_id": chunk.tenant_id,
            "knowledge_base_id": chunk.knowledge_base_id,
            "document_name": chunk.document_name,
            "source_uri": chunk.source_uri,
            "heading_path_json": self._json(chunk.heading_path),
            "section_title": chunk.section_title or "",
            "page_start": chunk.page_start or 0,
            "page_end": chunk.page_end or 0,
            "page_metadata_json": self._json(chunk.page_metadata),
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
            "embedding_provider": batch.provider,
            "embedding_model": batch.model,
            "embedding_version": batch.version,
            "embedding_dimension": batch.dimension,
            "created_at": created_at,
        }
        return TextNode(
            id_=chunk.chunk_id,
            text=chunk.content,
            embedding=list(vector),
            metadata=metadata,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=chunk.document_id)},
        )

    def _validate_search_input(
        self,
        *,
        query_str: str,
        query_embedding: RagQueryEmbedding,
        tenant_id: str,
        knowledge_base_id: str,
        similarity_top_k: int,
    ) -> None:
        if not isinstance(query_embedding, RagQueryEmbedding):
            raise MilvusStoreError("Query embedding contract is invalid.")
        if not isinstance(query_str, str) or not query_str.strip() or len(query_str) > 2000:
            raise MilvusStoreError("Milvus query text is invalid.")
        for name, value in (("tenant_id", tenant_id), ("knowledge_base_id", knowledge_base_id)):
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise MilvusStoreError(f"{name} must not be blank.")
        if (
            isinstance(similarity_top_k, bool)
            or not isinstance(similarity_top_k, int)
            or similarity_top_k < 1
            or similarity_top_k > RAG_TOP_K_MAX
        ):
            raise MilvusStoreError("similarity_top_k is outside the frozen range.")
        for name, value in (
            ("provider", query_embedding.provider),
            ("model", query_embedding.model),
            ("version", query_embedding.version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise MilvusStoreError(f"Query embedding {name} must not be blank.")
        if query_embedding.dimension != self.dimension:
            raise MilvusStoreError("Query embedding dimension does not match the collection.")
        self._validate_vector(query_embedding.vector)

    def _validate_vector(self, vector: Any) -> None:
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise MilvusStoreError("Query embedding vector must be numeric.")
        if len(vector) != self.dimension:
            raise MilvusStoreError("Query embedding vector has the wrong dimension.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise MilvusStoreError("Query embedding vector contains an invalid value.")

    def _normalize_search_result(
        self,
        result: Any,
        *,
        query_embedding: RagQueryEmbedding,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> tuple[RagHybridSearchHit, ...]:
        if result is None:
            raise MilvusStoreError("Milvus returned no query result contract.")
        ids = self._result_sequence(getattr(result, "ids", None), "ids")
        nodes = self._result_sequence(getattr(result, "nodes", None), "nodes")
        similarities = self._result_sequence(
            getattr(result, "similarities", None), "similarities"
        )
        if not (len(ids) == len(nodes) == len(similarities)):
            raise MilvusStoreError("Milvus query result lengths do not match.")
        if any(
            not isinstance(chunk_id, str)
            or not chunk_id.strip()
            or len(chunk_id) > 128
            for chunk_id in ids
        ):
            raise MilvusStoreError("Milvus query result IDs are invalid.")
        if len(set(ids)) != len(ids):
            raise MilvusStoreError("Milvus query result IDs must be unique.")

        hits: list[RagHybridSearchHit] = []
        for chunk_id, node, score in zip(ids, nodes, similarities, strict=True):
            hits.append(
                self._normalize_hit(
                    chunk_id,
                    node,
                    score,
                    query_embedding=query_embedding,
                    tenant_id=tenant_id,
                    knowledge_base_id=knowledge_base_id,
                )
            )
        return tuple(hits)

    @staticmethod
    def _result_sequence(value: Any, name: str) -> tuple[Any, ...]:
        if value is None or isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise MilvusStoreError(f"Milvus query result {name} is invalid.")
        return tuple(value)

    def _normalize_hit(
        self,
        chunk_id: Any,
        node: Any,
        score: Any,
        *,
        query_embedding: RagQueryEmbedding,
        tenant_id: str,
        knowledge_base_id: str,
    ) -> RagHybridSearchHit:
        chunk_id = self._required_string(chunk_id, "chunk_id", maximum=128)
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
            raise MilvusStoreError("Milvus query result score is invalid.")
        try:
            content = node.get_content(metadata_mode=MetadataMode.NONE)
            metadata = node.metadata
        except (AttributeError, TypeError):
            raise MilvusStoreError("Milvus query result node is invalid.") from None
        content = self._required_string(content, "content", maximum=16000)
        if not isinstance(metadata, Mapping):
            raise MilvusStoreError("Milvus query result metadata is invalid.")

        document_id = self._required_string(metadata.get("document_id"), "document_id", 128)
        hit_tenant = self._required_string(metadata.get("tenant_id"), "tenant_id", 128)
        hit_kb = self._required_string(
            metadata.get("knowledge_base_id"), "knowledge_base_id", 128
        )
        if hit_tenant != tenant_id or hit_kb != knowledge_base_id:
            raise MilvusStoreIsolationError("Milvus query result violated tenant scope.")
        document_name = self._required_string(
            metadata.get("document_name"), "document_name", 512
        )
        self._required_string(metadata.get("source_uri"), "source_uri")
        self._required_string(metadata.get("content_hash"), "content_hash")
        self._required_string(metadata.get("created_at"), "created_at")
        self._validate_json_object(metadata.get("page_metadata_json"))
        self._nonnegative_int(metadata.get("chunk_index"), "chunk_index")

        if (
            metadata.get("embedding_provider") != query_embedding.provider
            or metadata.get("embedding_model") != query_embedding.model
            or metadata.get("embedding_version") != query_embedding.version
            or metadata.get("embedding_dimension") != query_embedding.dimension
        ):
            raise MilvusStoreError("Milvus hit embedding identity is incompatible.")

        heading_path = self._heading_path(metadata.get("heading_path_json"))
        raw_section_title = metadata.get("section_title")
        if not isinstance(raw_section_title, str):
            raise MilvusStoreError("Milvus section_title metadata is invalid.")
        if len(raw_section_title) > 512:
            raise MilvusStoreError("Milvus section_title metadata is too long.")
        page_start = self._page(metadata.get("page_start"), "page_start")
        page_end = self._page(metadata.get("page_end"), "page_end")
        if page_start is not None and page_end is not None and page_end < page_start:
            raise MilvusStoreError("Milvus page metadata is invalid.")
        return RagHybridSearchHit(
            chunk_id=chunk_id,
            document_id=document_id,
            tenant_id=hit_tenant,
            knowledge_base_id=hit_kb,
            document_name=document_name,
            content=content,
            score=float(score),
            heading_path=heading_path,
            section_title=raw_section_title or None,
            page_start=page_start,
            page_end=page_end,
        )

    @staticmethod
    def _required_string(value: Any, name: str, maximum: int | None = None) -> str:
        if not isinstance(value, str) or not value.strip():
            raise MilvusStoreError(f"Milvus {name} metadata is invalid.")
        if maximum is not None and len(value) > maximum:
            raise MilvusStoreError(f"Milvus {name} metadata is too long.")
        return value

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MilvusStoreError(f"Milvus {name} metadata is invalid.")
        return value

    def _page(self, value: Any, name: str) -> int | None:
        page = self._nonnegative_int(value, name)
        return page or None

    @staticmethod
    def _heading_path(value: Any) -> tuple[str, ...]:
        if not isinstance(value, str):
            raise MilvusStoreError("Milvus heading_path_json metadata is invalid.")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            raise MilvusStoreError("Milvus heading_path_json metadata is invalid.") from None
        if (
            not isinstance(parsed, list)
            or len(parsed) > 16
            or any(not isinstance(item, str) or not item.strip() or len(item) > 512 for item in parsed)
        ):
            raise MilvusStoreError("Milvus heading_path_json metadata is invalid.")
        return tuple(parsed)

    @staticmethod
    def _validate_json_object(value: Any) -> None:
        if not isinstance(value, str):
            raise MilvusStoreError("Milvus page_metadata_json metadata is invalid.")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            raise MilvusStoreError("Milvus page_metadata_json metadata is invalid.") from None
        if not isinstance(parsed, dict):
            raise MilvusStoreError("Milvus page_metadata_json metadata is invalid.")

    async def _close_store(self, store: Any) -> None:
        failures: list[str] = []
        async_client = getattr(store, "_async_milvusclient", None)
        if async_client is not None:
            close = getattr(async_client, "close", None)
            if close is not None:
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except (Exception, asyncio.CancelledError):
                    failures.append("async_client")
        sync_client = getattr(store, "_milvusclient", None)
        if sync_client is not None:
            close = getattr(sync_client, "close", None)
            if close is not None:
                try:
                    await asyncio.to_thread(close)
                except (Exception, asyncio.CancelledError):
                    failures.append("sync_client")
        if failures:
            labels = ", ".join(failures)
            raise MilvusStoreError(f"Failed to close Milvus resources: {labels}.")

    @staticmethod
    def _close_store_from_init_thread(store: Any) -> None:
        async_client = getattr(store, "_async_milvusclient", None)
        if async_client is not None:
            close = getattr(async_client, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    asyncio.run(result)
        sync_client = getattr(store, "_milvusclient", None)
        if sync_client is not None:
            close = getattr(sync_client, "close", None)
            if close is not None:
                close()

    def _utc_timestamp(self) -> str:
        value = self._now()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
