from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Protocol

from superbiz_agent.config import RAG_TOP_K_MAX, Settings
from superbiz_agent.persistence.repositories.rag import (
    RagDocumentRepository,
    RagKnowledgeBaseContractError,
    RagKnowledgeBaseRepository,
    RagPersistenceError,
)
from superbiz_agent.rag.embedding import (
    RagEmbeddingIdentity,
    RagEmbeddingError,
    RagEmbeddingService,
    RagQueryEmbedding,
)
from superbiz_agent.rag.models import (
    RagChunk,
    RagRetrievalMode,
    RagRetrievalRequest,
    RagRetrievalResult,
)
from superbiz_agent.tools.fixtures import query_internal_docs_result

if TYPE_CHECKING:
    from superbiz_agent.rag.milvus_store import RagHybridSearchHit


_SAFE_NEXT_ACTIONS = (
    "use_alternative_tool",
    "search_memory_for_historical_reference",
    "degrade_with_user_friendly_error",
)


class RagRetrievalUnavailableError(RuntimeError):
    status_code = 503
    code = "rag_retrieval_unavailable"
    retryable = True

    def __init__(self) -> None:
        super().__init__("Internal document retrieval is temporarily unavailable.")


class RagRetrievalContractError(RuntimeError):
    code = "rag_retrieval_contract_error"
    retryable = False
    retryable_by_model = False
    allowed_next_actions = _SAFE_NEXT_ACTIONS

    def __init__(self) -> None:
        super().__init__("Internal document retrieval returned an invalid result.")


class RagRetrievalIsolationError(PermissionError):
    code = "rag_retrieval_isolation_error"
    retryable = False
    retryable_by_model = False
    allowed_next_actions = _SAFE_NEXT_ACTIONS

    def __init__(self) -> None:
        super().__init__("Internal document retrieval scope validation failed.")


class RagKnowledgeBaseNotConfiguredError(RuntimeError):
    code = "knowledge_base_not_configured"
    retryable = False
    retryable_by_model = False
    allowed_next_actions = (
        *_SAFE_NEXT_ACTIONS,
        "ask_admin_to_configure_knowledge_base",
    )

    def __init__(self) -> None:
        super().__init__("No active default knowledge base is configured for this tenant.")


class RagRetrievalDisabledError(RuntimeError):
    code = "rag_retrieval_disabled"
    retryable = False
    retryable_by_model = False
    allowed_next_actions = _SAFE_NEXT_ACTIONS

    def __init__(self) -> None:
        super().__init__("Internal document retrieval is disabled.")


class RagRetrievalService(Protocol):
    async def search(self, request: RagRetrievalRequest) -> RagRetrievalResult: ...


class DefaultKnowledgeBaseResolver(Protocol):
    async def resolve(self, tenant_id: str) -> str | None: ...


class DocumentStatusRepository(Protocol):
    async def get_document_statuses(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_ids: Sequence[str],
    ) -> Mapping[str, str]: ...


class HybridChunkStore(Protocol):
    async def hybrid_search(self, **kwargs: object) -> Sequence["RagHybridSearchHit"]: ...

    async def search(self, **kwargs: object) -> Sequence["RagHybridSearchHit"]: ...


class RepositoryDefaultKnowledgeBaseResolver:
    def __init__(self, repository: RagKnowledgeBaseRepository) -> None:
        self._repository = repository

    async def resolve(self, tenant_id: str) -> str | None:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise RagRetrievalContractError()
        try:
            record = await self._repository.get_default_for_tenant(tenant_id=tenant_id)
        except RagKnowledgeBaseContractError:
            raise RagRetrievalContractError() from None
        except RagPersistenceError:
            raise RagRetrievalUnavailableError() from None
        except Exception:
            raise RagRetrievalUnavailableError() from None
        if record is None:
            return None
        if (
            record.tenant_id != tenant_id
            or record.status != "active"
            or record.is_default is not True
            or not record.id.strip()
        ):
            raise RagRetrievalContractError()
        return record.id


class MilvusRagRetrievalService:
    def __init__(
        self,
        *,
        settings: Settings,
        knowledge_base_resolver: DefaultKnowledgeBaseResolver,
        document_repository: DocumentStatusRepository | RagDocumentRepository,
        embedding_service: RagEmbeddingService,
        chunk_store: HybridChunkStore,
        retrieval_mode: RagRetrievalMode = RagRetrievalMode.HYBRID,
    ) -> None:
        if not isinstance(retrieval_mode, RagRetrievalMode):
            raise ValueError("retrieval_mode is invalid")
        self._settings = settings
        self._knowledge_bases = knowledge_base_resolver
        self._documents = document_repository
        self._embedding = embedding_service
        self._chunks = chunk_store
        self._retrieval_mode = retrieval_mode

    async def search(self, request: RagRetrievalRequest) -> RagRetrievalResult:
        query = request.query
        tenant_id = request.scope.tenant_id
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 2000
            or not isinstance(tenant_id, str)
            or not tenant_id.strip()
        ):
            raise RagRetrievalContractError()
        if (
            self._settings.rag_hybrid_top_k > RAG_TOP_K_MAX
            or self._settings.rag_final_top_k > RAG_TOP_K_MAX
        ):
            raise RagRetrievalContractError()

        try:
            knowledge_base_id = await self._knowledge_bases.resolve(tenant_id)
        except (
            RagKnowledgeBaseNotConfiguredError,
            RagRetrievalContractError,
            RagRetrievalUnavailableError,
        ):
            raise
        except Exception:
            raise RagRetrievalUnavailableError() from None
        if knowledge_base_id is None:
            raise RagKnowledgeBaseNotConfiguredError()
        if not isinstance(knowledge_base_id, str) or not knowledge_base_id.strip():
            raise RagRetrievalContractError()

        embedding_identity = RagEmbeddingIdentity(
            provider=self._settings.rag_embedding_provider,
            model=self._settings.rag_embedding_model,
            version=(self._settings.rag_embedding_version or self._settings.rag_embedding_model),
            dimension=self._settings.rag_embedding_dimension,
        )
        query_embedding = None
        if self._retrieval_mode is not RagRetrievalMode.BM25:
            try:
                query_embedding = await self._embedding.embed_query(query)
            except RagEmbeddingError:
                raise RagRetrievalContractError() from None
            except Exception:
                raise RagRetrievalUnavailableError() from None
            self._validate_query_embedding(query_embedding)

        from superbiz_agent.rag.milvus_store import (
            MilvusStoreError,
            MilvusStoreIsolationError,
            RagHybridSearchHit,
        )

        try:
            search_arguments = {
                "query_str": query,
                "query_embedding": query_embedding,
                "tenant_id": tenant_id,
                "knowledge_base_id": knowledge_base_id,
                "similarity_top_k": self._settings.rag_hybrid_top_k,
            }
            if self._retrieval_mode is RagRetrievalMode.HYBRID:
                hits = await self._chunks.hybrid_search(**search_arguments)
            else:
                hits = await self._chunks.search(
                    mode=self._retrieval_mode,
                    embedding_identity=embedding_identity,
                    **search_arguments,
                )
        except MilvusStoreIsolationError:
            raise RagRetrievalIsolationError() from None
        except MilvusStoreError:
            raise RagRetrievalContractError() from None
        except Exception:
            raise RagRetrievalUnavailableError() from None
        if (
            isinstance(hits, (str, bytes))
            or not isinstance(hits, Sequence)
            or len(hits) > self._settings.rag_hybrid_top_k
            or any(not isinstance(hit, RagHybridSearchHit) for hit in hits)
        ):
            raise RagRetrievalContractError()
        if not hits:
            return RagRetrievalResult()

        self._validate_hits(hits, tenant_id, knowledge_base_id)

        document_ids = tuple(dict.fromkeys(hit.document_id for hit in hits))
        try:
            statuses = await self._documents.get_document_statuses(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                document_ids=document_ids,
            )
        except RagPersistenceError:
            raise RagRetrievalUnavailableError() from None
        except Exception:
            raise RagRetrievalUnavailableError() from None
        allowed_statuses = {"active", "indexing", "failed", "archived"}
        if (
            not isinstance(statuses, Mapping)
            or set(statuses) != set(document_ids)
            or any(
                not isinstance(status, str) or status not in allowed_statuses
                for status in statuses.values()
            )
        ):
            raise RagRetrievalContractError()

        active_hits = [hit for hit in hits if statuses[hit.document_id] == "active"]
        final_hits = active_hits[: self._settings.rag_final_top_k]
        return RagRetrievalResult(
            chunks=tuple(self._to_rag_chunk(hit) for hit in final_hits)
        )

    def _validate_query_embedding(self, embedding: RagQueryEmbedding) -> None:
        expected_version = (
            self._settings.rag_embedding_version or self._settings.rag_embedding_model
        )
        if not isinstance(embedding, RagQueryEmbedding):
            raise RagRetrievalContractError()
        vector = embedding.vector
        if (
            embedding.provider != self._settings.rag_embedding_provider
            or embedding.model != self._settings.rag_embedding_model
            or embedding.version != expected_version
            or embedding.dimension != self._settings.rag_embedding_dimension
            or isinstance(vector, (str, bytes))
            or not isinstance(vector, Sequence)
            or len(vector) != self._settings.rag_embedding_dimension
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in vector
            )
        ):
            raise RagRetrievalContractError()

    @staticmethod
    def _validate_hits(
        hits: Sequence["RagHybridSearchHit"], tenant_id: str, knowledge_base_id: str
    ) -> None:
        seen_ids: set[str] = set()
        for hit in hits:
            if hit.tenant_id != tenant_id or hit.knowledge_base_id != knowledge_base_id:
                raise RagRetrievalIsolationError()
            if (
                not isinstance(hit.chunk_id, str)
                or not hit.chunk_id.strip()
                or len(hit.chunk_id) > 128
                or hit.chunk_id in seen_ids
                or not isinstance(hit.document_id, str)
                or not hit.document_id.strip()
                or len(hit.document_id) > 128
                or not isinstance(hit.document_name, str)
                or not hit.document_name.strip()
                or len(hit.document_name) > 512
                or not isinstance(hit.content, str)
                or not hit.content.strip()
                or len(hit.content) > 16000
                or isinstance(hit.score, bool)
                or not isinstance(hit.score, Real)
                or not math.isfinite(float(hit.score))
            ):
                raise RagRetrievalContractError()
            seen_ids.add(hit.chunk_id)

    def _to_rag_chunk(self, hit: "RagHybridSearchHit") -> RagChunk:
        return RagChunk(
            id=hit.chunk_id,
            source=hit.document_name,
            content=hit.content,
            score=hit.score,
            metadata={
                "documentId": hit.document_id,
                "documentName": hit.document_name,
                "knowledgeBaseId": hit.knowledge_base_id,
                "headingPath": list(hit.heading_path),
                "sectionTitle": hit.section_title,
                "pageStart": hit.page_start,
                "pageEnd": hit.page_end,
                "retrievalSource": self._retrieval_mode.value,
                "confidence": "unavailable",
                "evidenceType": "untrusted_retrieved_document",
            },
        )


class FixtureRagRetrievalService:
    """Deterministic local/test-only replacement for the real Milvus retrieval service."""

    async def search(self, request: RagRetrievalRequest) -> RagRetrievalResult:
        try:
            raw_result = query_internal_docs_result(request.query)
            if not isinstance(raw_result, Mapping):
                raise RagRetrievalContractError()
            if raw_result.get("status") != "ok":
                return RagRetrievalResult(message=raw_result.get("message"))
            raw_chunks = raw_result.get("chunks")
            if (
                isinstance(raw_chunks, (str, bytes))
                or not isinstance(raw_chunks, Sequence)
                or any(not isinstance(chunk, Mapping) for chunk in raw_chunks)
            ):
                raise RagRetrievalContractError()
            return RagRetrievalResult(
                chunks=tuple(
                    RagChunk(
                        id=chunk.get("id"),  # type: ignore[arg-type]
                        source=chunk.get("source"),  # type: ignore[arg-type]
                        content=chunk.get("content"),  # type: ignore[arg-type]
                        score=chunk.get("score"),  # type: ignore[arg-type]
                        metadata={
                            key: value
                            for key, value in chunk.items()
                            if key not in {"id", "ref", "source", "content", "score"}
                        },
                    )
                    for chunk in raw_chunks
                )
            )
        except RagRetrievalContractError:
            raise
        except Exception:
            raise RagRetrievalContractError() from None


class UnavailableRagRetrievalService:
    def __init__(self, message: str) -> None:
        self.message = message

    async def search(self, request: RagRetrievalRequest) -> RagRetrievalResult:
        del request
        raise RagRetrievalDisabledError()
