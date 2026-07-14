from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real

from superbiz_agent.config import Settings
from superbiz_agent.persistence.repositories.rag import (
    RagClaimLostError,
    RagDocumentArchivedError,
    RagDocumentClaimStatus,
    RagDocumentRepository,
    RagKnowledgeBaseRepository,
    RagKnowledgeBaseNotActiveError,
    RagPersistenceError,
)
from superbiz_agent.rag.chunking import RagDocumentChunker
from superbiz_agent.rag.embedding import RagEmbeddingBatch, RagEmbeddingService
from superbiz_agent.rag.milvus_store import RagChunkStore
from superbiz_agent.rag.models import (
    RagDocumentContentType,
    RagIngestionRequest,
    RagIngestionResult,
    RagIngestionStatus,
    RagSourceDocument,
)


_CONTENT_TYPE_ALIASES = {
    "markdown": RagDocumentContentType.MARKDOWN,
    "md": RagDocumentContentType.MARKDOWN,
    "text/markdown": RagDocumentContentType.MARKDOWN,
    "txt": RagDocumentContentType.TXT,
    "text": RagDocumentContentType.TXT,
    "text/plain": RagDocumentContentType.TXT,
    "parsed_text": RagDocumentContentType.PARSED_TEXT,
    "parsed-text": RagDocumentContentType.PARSED_TEXT,
}

_FAILURES: Mapping[str, tuple[str, str]] = {
    "normalize": ("invalid_document", "Document input is invalid."),
    "knowledge_base": (
        "knowledge_base_not_active",
        "Knowledge base is not active for tenant.",
    ),
    "knowledge_base_store": (
        "knowledge_base_store_error",
        "Knowledge base validation failed.",
    ),
    "archived": ("document_archived", "Archived document cannot be ingested."),
    "persistence": ("rag_persistence_error", "RAG persistence operation failed."),
    "claim": ("document_claim_failed", "Document ingestion claim failed."),
    "chunk": ("chunking_failed", "Document chunking failed."),
    "empty_chunks": ("empty_chunks", "Document chunking produced no chunks."),
    "refresh": ("claim_refresh_failed", "Document ingestion claim refresh failed."),
    "embed": ("embedding_failed", "Document embedding failed."),
    "store": ("chunk_store_failed", "Document chunk storage failed."),
    "active": ("mark_active_failed", "Document activation failed."),
    "claim_lost": ("claim_lost", "Document ingestion claim was lost."),
}


class RagIngestionError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        code: str,
        message: str,
        document_id: str | None = None,
    ) -> None:
        super().__init__(message[:1000])
        self.stage = stage
        self.code = code
        self.document_id = document_id
        self.safe_message = message[:1000]


def canonicalize_document_content(
    content: str | bytes,
    content_type: RagDocumentContentType | str,
) -> tuple[str, RagDocumentContentType]:
    canonical_type = _canonical_content_type(content_type)
    if isinstance(content, bytes):
        try:
            decoded = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("document content must be valid UTF-8") from exc
    elif isinstance(content, str):
        decoded = content
    else:
        raise TypeError("document content must be a string or bytes")
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("document content must not be blank")
    return normalized, canonical_type


def document_content_hash(content: str, content_type: RagDocumentContentType) -> str:
    payload = json.dumps(
        {"content": content, "content_type": content_type.value},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_content_type(
    content_type: RagDocumentContentType | str,
) -> RagDocumentContentType:
    if isinstance(content_type, RagDocumentContentType):
        return content_type
    if not isinstance(content_type, str):
        raise TypeError("content_type must be a string or RagDocumentContentType")
    resolved = _CONTENT_TYPE_ALIASES.get(content_type.strip().lower())
    if resolved is None:
        raise ValueError("unsupported document content type")
    return resolved


class RagIngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        knowledge_base_repository: RagKnowledgeBaseRepository,
        document_repository: RagDocumentRepository,
        chunker: RagDocumentChunker,
        embedding_service: RagEmbeddingService,
        chunk_store: RagChunkStore,
    ) -> None:
        self._settings = settings
        self._knowledge_bases = knowledge_base_repository
        self._documents = document_repository
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._chunk_store = chunk_store

    async def ingest_sync(self, request: RagIngestionRequest) -> RagIngestionResult:
        try:
            content, content_type = canonicalize_document_content(
                request.content, request.content_type
            )
            content_hash = document_content_hash(content, content_type)
        except (TypeError, ValueError):
            raise self._error("normalize") from None

        try:
            knowledge_base = await self._knowledge_bases.get_active_for_tenant(
                tenant_id=request.tenant_id,
                knowledge_base_id=request.knowledge_base_id,
            )
        except RagPersistenceError:
            raise self._error("knowledge_base_store") from None
        except Exception:
            raise self._error("knowledge_base_store") from None
        if knowledge_base is None:
            raise self._error("knowledge_base")

        try:
            claim = await self._documents.claim_for_ingestion(
                tenant_id=request.tenant_id,
                knowledge_base_id=request.knowledge_base_id,
                source_uri=request.source_uri,
                document_name=request.document_name,
                content_type=content_type.value,
                parser=request.parser,
                parser_version=request.parser_version,
                embedding_provider=self._settings.rag_embedding_provider,
                embedding_model=self._settings.rag_embedding_model,
                embedding_version=self._settings.rag_embedding_version
                or self._settings.rag_embedding_model,
                embedding_dimension=self._settings.rag_embedding_dimension,
                content_hash=content_hash,
                created_by=request.created_by,
                stale_after_seconds=self._settings.rag_ingestion_stale_after_seconds,
            )
        except RagKnowledgeBaseNotActiveError:
            raise self._error("knowledge_base") from None
        except RagDocumentArchivedError:
            raise self._error("archived") from None
        except RagPersistenceError:
            raise self._error("persistence") from None
        except Exception:
            raise self._error("claim") from None

        if claim.status is RagDocumentClaimStatus.DUPLICATE_SKIPPED:
            return RagIngestionResult(
                status=RagIngestionStatus.DUPLICATE_SKIPPED,
                document_id=claim.document_id,
                content_hash=claim.content_hash,
                chunk_count=claim.chunk_count,
            )
        if claim.status is RagDocumentClaimStatus.IN_PROGRESS:
            return RagIngestionResult(
                status=RagIngestionStatus.IN_PROGRESS,
                document_id=claim.document_id,
                content_hash=claim.content_hash,
            )
        if claim.status is not RagDocumentClaimStatus.CLAIMED or not claim.claim_token:
            raise self._error("claim", document_id=claim.document_id)

        document_id = claim.document_id
        claim_token = claim.claim_token
        try:
            source = RagSourceDocument(
                document_id=document_id,
                tenant_id=request.tenant_id,
                knowledge_base_id=request.knowledge_base_id,
                document_name=request.document_name,
                source_uri=request.source_uri,
                content=content,
                content_type=content_type,
                page_start=request.page_start,
                page_end=request.page_end,
                page_metadata=dict(request.page_metadata),
            )
            chunks = tuple(await asyncio.to_thread(self._chunker.chunk, source))
        except asyncio.CancelledError:
            await self._fail_if_owned(request, document_id, claim_token, "chunk")
            raise
        except Exception:
            await self._fail_if_owned(request, document_id, claim_token, "chunk")
            raise self._error("chunk", document_id=document_id) from None
        if not chunks:
            await self._fail_if_owned(request, document_id, claim_token, "empty_chunks")
            raise self._error("empty_chunks", document_id=document_id)

        await self._refresh_or_fail(request, document_id, claim_token)
        try:
            embedding_batch = await self._embedding_service.embed_documents(
                tuple(chunk.content for chunk in chunks)
            )
            self._validate_embedding_batch(embedding_batch, len(chunks))
        except asyncio.CancelledError:
            await self._fail_if_owned(request, document_id, claim_token, "embed")
            raise
        except Exception:
            await self._fail_if_owned(request, document_id, claim_token, "embed")
            raise self._error("embed", document_id=document_id) from None

        await self._refresh_or_fail(request, document_id, claim_token)
        try:
            await self._chunk_store.upsert(chunks, embedding_batch)
        except asyncio.CancelledError:
            await self._fail_if_owned(request, document_id, claim_token, "store")
            raise
        except Exception:
            await self._fail_if_owned(request, document_id, claim_token, "store")
            raise self._error("store", document_id=document_id) from None

        try:
            await self._documents.mark_active(
                tenant_id=request.tenant_id,
                document_id=document_id,
                claim_token=claim_token,
                chunk_count=len(chunks),
            )
        except RagClaimLostError:
            raise self._error("claim_lost", document_id=document_id) from None
        except asyncio.CancelledError:
            await self._fail_if_owned(request, document_id, claim_token, "active")
            raise
        except Exception:
            await self._fail_if_owned(request, document_id, claim_token, "active")
            raise self._error("active", document_id=document_id) from None

        return RagIngestionResult(
            status=RagIngestionStatus.INDEXED,
            document_id=document_id,
            content_hash=claim.content_hash,
            chunk_count=len(chunks),
        )

    async def _refresh_or_fail(
        self, request: RagIngestionRequest, document_id: str, claim_token: str
    ) -> None:
        try:
            await self._documents.refresh_claim(
                tenant_id=request.tenant_id,
                document_id=document_id,
                claim_token=claim_token,
            )
        except RagClaimLostError:
            raise self._error("claim_lost", document_id=document_id) from None
        except asyncio.CancelledError:
            await self._fail_if_owned(request, document_id, claim_token, "refresh")
            raise
        except Exception:
            await self._fail_if_owned(request, document_id, claim_token, "refresh")
            raise self._error("refresh", document_id=document_id) from None

    async def _fail_if_owned(
        self,
        request: RagIngestionRequest,
        document_id: str,
        claim_token: str,
        failure: str,
    ) -> None:
        code, message = _FAILURES[failure]
        cleanup = asyncio.create_task(
            self._documents.mark_failed(
                tenant_id=request.tenant_id,
                document_id=document_id,
                claim_token=claim_token,
                failure_code=code,
                sanitized_message=message,
            )
        )
        try:
            await asyncio.shield(cleanup)
        except Exception:
            pass

    def _validate_embedding_batch(self, batch: RagEmbeddingBatch, expected_count: int) -> None:
        if len(batch.vectors) != expected_count:
            raise ValueError("embedding count mismatch")
        expected_version = (
            self._settings.rag_embedding_version or self._settings.rag_embedding_model
        )
        if (
            batch.provider != self._settings.rag_embedding_provider
            or batch.model != self._settings.rag_embedding_model
            or batch.version != expected_version
            or batch.dimension != self._settings.rag_embedding_dimension
        ):
            raise ValueError("embedding metadata mismatch")
        for vector in batch.vectors:
            if len(vector) != batch.dimension:
                raise ValueError("embedding vector dimension mismatch")
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("embedding vector contains a non-numeric value")
                if not math.isfinite(float(value)):
                    raise ValueError("embedding vector contains a non-finite value")

    @staticmethod
    def _error(stage: str, document_id: str | None = None) -> RagIngestionError:
        code, message = _FAILURES[stage]
        return RagIngestionError(
            stage=stage,
            code=code,
            message=message,
            document_id=document_id,
        )
