from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import traceback
from dataclasses import dataclass
from typing import Any

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.persistence.repositories.rag import (
    RagClaimLostError,
    RagDocumentArchivedError,
    RagDocumentClaim,
    RagDocumentClaimStatus,
    RagKnowledgeBaseNotActiveError,
    RagPersistenceError,
)
from superbiz_agent.rag.embedding import RagEmbeddingBatch
from superbiz_agent.rag.ingestion import (
    RagIngestionError,
    RagIngestionService,
    canonicalize_document_content,
    document_content_hash,
)
from superbiz_agent.rag.models import (
    RagDocumentContentType,
    RagIngestionRequest,
    RagIngestionStatus,
    RagPreparedChunk,
)


CONTENT = "private-body-c5d128"
SECRET = "secret-key-c5d128"
DOCUMENT_ID = "document-stable"
CLAIM_TOKEN = "claim-token"


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        rag_embedding_provider="test-provider",
        rag_embedding_model="test-model",
        rag_embedding_version="test-v1",
        rag_embedding_dimension=3,
        rag_ingestion_stale_after_seconds=30,
    )


def _request(**overrides: Any) -> RagIngestionRequest:
    values = {
        "tenant_id": "tenant-a",
        "knowledge_base_id": "kb-a",
        "document_name": "runbook.md",
        "source_uri": "trusted://runbook.md",
        "content": CONTENT,
        "content_type": "text/plain",
        "created_by": "admin-a",
        "parser": "llama-index",
        "parser_version": "0.14.23",
        "page_metadata": {"page": 1},
    }
    values.update(overrides)
    return RagIngestionRequest(**values)


def _claim(
    status: RagDocumentClaimStatus = RagDocumentClaimStatus.CLAIMED,
    *,
    document_id: str = DOCUMENT_ID,
    chunk_count: int = 0,
) -> RagDocumentClaim:
    content, content_type = canonicalize_document_content(CONTENT, "text/plain")
    return RagDocumentClaim(
        status=status,
        document_id=document_id,
        content_hash=document_content_hash(content, content_type),
        chunk_count=chunk_count,
        claim_token=CLAIM_TOKEN if status is RagDocumentClaimStatus.CLAIMED else None,
    )


@dataclass
class _KnowledgeBase:
    id: str = "kb-a"
    tenant_id: str = "tenant-a"


class _KnowledgeBaseRepository:
    def __init__(self, events: list[str], active: bool = True, error: BaseException | None = None):
        self.events = events
        self.active = active
        self.error = error

    async def get_active_for_tenant(self, **kwargs: Any) -> _KnowledgeBase | None:
        self.events.append("kb")
        if self.error:
            raise self.error
        if kwargs != {"tenant_id": "tenant-a", "knowledge_base_id": "kb-a"}:
            return None
        return _KnowledgeBase() if self.active else None


class _DocumentRepository:
    def __init__(
        self,
        events: list[str],
        claims: list[RagDocumentClaim] | None = None,
    ) -> None:
        self.events = events
        self.claims = list(claims or [_claim()])
        self.claim_kwargs: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self.claim_error: BaseException | None = None
        self.refresh_error_at: dict[int, BaseException] = {}
        self.active_error: BaseException | None = None
        self.mark_failed_error: BaseException | None = None
        self.refresh_count = 0

    async def claim_for_ingestion(self, **kwargs: Any) -> RagDocumentClaim:
        self.events.append("claim")
        self.claim_kwargs.append(kwargs)
        if self.claim_error:
            raise self.claim_error
        return self.claims.pop(0)

    async def refresh_claim(self, **kwargs: Any) -> None:
        self.refresh_count += 1
        self.events.append(f"refresh-{self.refresh_count}")
        error = self.refresh_error_at.get(self.refresh_count)
        if error:
            raise error

    async def mark_active(self, **kwargs: Any) -> None:
        self.events.append("active")
        if self.active_error:
            raise self.active_error

    async def mark_failed(self, **kwargs: Any) -> None:
        self.events.append("failed")
        self.failed.append(kwargs)
        await asyncio.sleep(0)
        if self.mark_failed_error:
            raise self.mark_failed_error


class _Chunker:
    def __init__(
        self,
        events: list[str],
        *,
        error: Exception | None = None,
        empty: bool = False,
    ) -> None:
        self.events = events
        self.error = error
        self.empty = empty
        self.sources: list[Any] = []

    def chunk(self, source: Any) -> tuple[RagPreparedChunk, ...]:
        self.events.append("chunk")
        self.sources.append(source)
        if self.error:
            raise self.error
        if self.empty:
            return ()
        content = str(source.content)
        return (
            RagPreparedChunk(
                chunk_id=f"{source.document_id}:000000",
                tenant_id=source.tenant_id,
                knowledge_base_id=source.knowledge_base_id,
                document_id=source.document_id,
                document_name=source.document_name,
                source_uri=source.source_uri,
                heading_path=(),
                section_title=None,
                page_start=source.page_start,
                page_end=source.page_end,
                page_metadata=dict(source.page_metadata),
                chunk_index=0,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                content=content,
            ),
        )


class _EmbeddingService:
    def __init__(
        self,
        events: list[str],
        error: BaseException | None = None,
        batch: RagEmbeddingBatch | None = None,
    ) -> None:
        self.events = events
        self.error = error
        self.batch = batch
        self.texts: list[tuple[str, ...]] = []

    async def embed_documents(self, texts: Any) -> RagEmbeddingBatch:
        self.events.append("embed")
        self.texts.append(tuple(texts))
        if self.error:
            raise self.error
        return self.batch or RagEmbeddingBatch(
            vectors=tuple((0.1, 0.2, 0.3) for _ in texts),
            provider="test-provider",
            model="test-model",
            version="test-v1",
            dimension=3,
        )


class _ChunkStore:
    def __init__(
        self,
        events: list[str],
        *,
        error: BaseException | None = None,
        fail_once_after_write: bool = False,
        cancel_after_write: bool = False,
    ) -> None:
        self.events = events
        self.error = error
        self.fail_once_after_write = fail_once_after_write
        self.cancel_after_write = cancel_after_write
        self.calls: list[tuple[str, ...]] = []
        self.persisted_ids: set[str] = set()

    async def upsert(self, chunks: Any, embedding_batch: Any) -> tuple[str, ...]:
        self.events.append("upsert")
        ids = tuple(chunk.chunk_id for chunk in chunks)
        self.calls.append(ids)
        self.persisted_ids.update(ids)
        if self.cancel_after_write:
            raise asyncio.CancelledError()
        if self.fail_once_after_write:
            self.fail_once_after_write = False
            raise RuntimeError(f"partial write {CONTENT} {SECRET}")
        if self.error:
            raise self.error
        return ids


def _service(
    events: list[str],
    *,
    knowledge_bases: _KnowledgeBaseRepository | None = None,
    documents: _DocumentRepository | None = None,
    chunker: _Chunker | None = None,
    embedding: _EmbeddingService | None = None,
    store: _ChunkStore | None = None,
) -> tuple[
    RagIngestionService,
    _DocumentRepository,
    _Chunker,
    _EmbeddingService,
    _ChunkStore,
]:
    documents = documents or _DocumentRepository(events)
    chunker = chunker or _Chunker(events)
    embedding = embedding or _EmbeddingService(events)
    store = store or _ChunkStore(events)
    return (
        RagIngestionService(
            settings=_settings(),
            knowledge_base_repository=knowledge_bases or _KnowledgeBaseRepository(events),
            document_repository=documents,
            chunker=chunker,
            embedding_service=embedding,
            chunk_store=store,
        ),
        documents,
        chunker,
        embedding,
        store,
    )


def test_canonical_content_type_crlf_and_deterministic_json_sha256() -> None:
    content, content_type = canonicalize_document_content("  alpha\r\nbeta\rgamma  ", "MD")

    assert content == "alpha\nbeta\ngamma"
    assert content_type is RagDocumentContentType.MARKDOWN
    expected_json = json.dumps(
        {"content": content, "content_type": "markdown"},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert document_content_hash(content, content_type) == hashlib.sha256(
        expected_json.encode()
    ).hexdigest()
    alias_content, alias_type = canonicalize_document_content(
        b"alpha\r\nbeta\rgamma", "text/markdown"
    )
    assert document_content_hash(alias_content, alias_type) == document_content_hash(
        content, content_type
    )


@pytest.mark.asyncio
async def test_indexed_flow_has_fenced_order_and_claim_document_id_source() -> None:
    events: list[str] = []
    service, documents, chunker, embedding, store = _service(events)

    result = await service.ingest_sync(_request(content="  first\r\nsecond  "))

    assert events == [
        "kb",
        "claim",
        "chunk",
        "refresh-1",
        "embed",
        "refresh-2",
        "upsert",
        "active",
    ]
    assert result.status is RagIngestionStatus.INDEXED
    assert result.document_id == DOCUMENT_ID
    assert result.chunk_count == 1
    assert chunker.sources[0].document_id == DOCUMENT_ID
    assert chunker.sources[0].content == "first\nsecond"
    assert embedding.texts == [("first\nsecond",)]
    assert store.calls == [(f"{DOCUMENT_ID}:000000",)]
    claim_kwargs = documents.claim_kwargs[0]
    assert claim_kwargs["content_type"] == "txt"
    assert claim_kwargs["embedding_dimension"] == 3
    assert claim_kwargs["stale_after_seconds"] == 30


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("claim_status", "result_status", "chunk_count"),
    [
        (RagDocumentClaimStatus.DUPLICATE_SKIPPED, RagIngestionStatus.DUPLICATE_SKIPPED, 7),
        (RagDocumentClaimStatus.IN_PROGRESS, RagIngestionStatus.IN_PROGRESS, 0),
    ],
)
async def test_duplicate_and_in_progress_have_no_later_side_effects(
    claim_status: RagDocumentClaimStatus,
    result_status: RagIngestionStatus,
    chunk_count: int,
) -> None:
    events: list[str] = []
    documents = _DocumentRepository(events, [_claim(claim_status, chunk_count=chunk_count)])
    service, _, _, _, _ = _service(events, documents=documents)

    result = await service.ingest_sync(_request())

    assert events == ["kb", "claim"]
    assert result.status is result_status
    assert result.chunk_count == (7 if result_status is RagIngestionStatus.DUPLICATE_SKIPPED else None)


@pytest.mark.asyncio
async def test_failed_or_stale_retry_reuses_document_and_chunk_ids() -> None:
    events: list[str] = []
    documents = _DocumentRepository(events, [_claim(), _claim()])
    first_chunker = _Chunker(events, error=RuntimeError("first failure"))
    store = _ChunkStore(events)
    first_service, _, _, _, _ = _service(
        events, documents=documents, chunker=first_chunker, store=store
    )
    with pytest.raises(RagIngestionError, match="Document chunking failed"):
        await first_service.ingest_sync(_request())

    retry_chunker = _Chunker(events)
    retry_service, _, _, _, _ = _service(
        events, documents=documents, chunker=retry_chunker, store=store
    )
    result = await retry_service.ingest_sync(_request())

    assert result.document_id == DOCUMENT_ID
    assert retry_chunker.sources[0].document_id == DOCUMENT_ID
    assert store.calls == [(f"{DOCUMENT_ID}:000000",)]
    assert documents.claim_kwargs[0]["content_hash"] == documents.claim_kwargs[1]["content_hash"]


@pytest.mark.asyncio
async def test_empty_chunks_marks_failed_without_embedding() -> None:
    events: list[str] = []
    service, documents, _, _, _ = _service(events, chunker=_Chunker(events, empty=True))

    with pytest.raises(RagIngestionError) as raised:
        await service.ingest_sync(_request())

    assert raised.value.code == "empty_chunks"
    assert events == ["kb", "claim", "chunk", "failed"]
    assert documents.failed[0]["failure_code"] == "empty_chunks"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["chunk", "embed", "refresh", "store", "active"])
async def test_stage_failures_best_effort_mark_failed(failure_stage: str) -> None:
    events: list[str] = []
    documents = _DocumentRepository(events)
    chunker = _Chunker(events, error=RuntimeError(SECRET)) if failure_stage == "chunk" else None
    embedding = (
        _EmbeddingService(events, error=RuntimeError(SECRET))
        if failure_stage == "embed"
        else None
    )
    store = _ChunkStore(events, error=RuntimeError(SECRET)) if failure_stage == "store" else None
    if failure_stage == "refresh":
        documents.refresh_error_at[1] = RuntimeError(SECRET)
    if failure_stage == "active":
        documents.active_error = RuntimeError(SECRET)
    service, documents, _, _, _ = _service(
        events,
        documents=documents,
        chunker=chunker,
        embedding=embedding,
        store=store,
    )

    with pytest.raises(RagIngestionError) as raised:
        await service.ingest_sync(_request())

    assert raised.value.stage == failure_stage
    assert events[-1] == "failed"
    assert len(documents.failed) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("location", "expected_tail"), [("refresh", "refresh-1"), ("active", "active")])
async def test_claim_lost_stops_and_never_marks_failed(location: str, expected_tail: str) -> None:
    events: list[str] = []
    documents = _DocumentRepository(events)
    if location == "refresh":
        documents.refresh_error_at[1] = RagClaimLostError()
    else:
        documents.active_error = RagClaimLostError()
    service, documents, _, _, _ = _service(events, documents=documents)

    with pytest.raises(RagIngestionError) as raised:
        await service.ingest_sync(_request())

    assert raised.value.code == "claim_lost"
    assert events[-1] == expected_tail
    assert documents.failed == []
    if location == "refresh":
        assert "embed" not in events
        assert "upsert" not in events


@pytest.mark.asyncio
async def test_second_refresh_claim_lost_prevents_upsert() -> None:
    events: list[str] = []
    documents = _DocumentRepository(events)
    documents.refresh_error_at[2] = RagClaimLostError()
    service, documents, _, _, store = _service(events, documents=documents)

    with pytest.raises(RagIngestionError, match="claim was lost"):
        await service.ingest_sync(_request())

    assert "embed" in events
    assert "upsert" not in events
    assert store.calls == []
    assert documents.failed == []


@pytest.mark.asyncio
async def test_partial_upsert_retry_uses_stable_ids() -> None:
    events: list[str] = []
    documents = _DocumentRepository(events, [_claim(), _claim()])
    store = _ChunkStore(events, fail_once_after_write=True)
    service, _, _, _, _ = _service(events, documents=documents, store=store)

    with pytest.raises(RagIngestionError, match="storage failed"):
        await service.ingest_sync(_request())
    result = await service.ingest_sync(_request())

    expected_ids = (f"{DOCUMENT_ID}:000000",)
    assert result.status is RagIngestionStatus.INDEXED
    assert store.calls == [expected_ids, expected_ids]
    assert store.persisted_ids == set(expected_ids)


@pytest.mark.asyncio
async def test_tenant_knowledge_base_rejection_precedes_claim_and_external_work() -> None:
    events: list[str] = []
    knowledge_bases = _KnowledgeBaseRepository(events, active=False)
    service, documents, _, _, store = _service(events, knowledge_bases=knowledge_bases)

    with pytest.raises(RagIngestionError) as raised:
        await service.ingest_sync(_request())

    assert raised.value.code == "knowledge_base_not_active"
    assert events == ["kb"]
    assert documents.claim_kwargs == []
    assert store.calls == []


@pytest.mark.asyncio
async def test_failure_metadata_and_error_do_not_leak_content_vector_or_secret() -> None:
    events: list[str] = []
    documents = _DocumentRepository(events)
    leaking_error = RuntimeError(f"{CONTENT} {SECRET} vector=[0.1, 0.2, 0.3]")
    service, documents, _, _, _ = _service(
        events,
        documents=documents,
        embedding=_EmbeddingService(events, error=leaking_error),
    )

    with pytest.raises(RagIngestionError) as raised:
        await service.ingest_sync(_request())

    persisted = json.dumps(documents.failed)
    visible_error = f"{raised.value} {raised.value.code} {raised.value.stage}"
    for sensitive in (CONTENT, SECRET, "[0.1, 0.2, 0.3]"):
        assert sensitive not in persisted
        assert sensitive not in visible_error
    assert len(documents.failed[0]["sanitized_message"]) <= 1000


@pytest.mark.asyncio
async def test_translated_error_suppresses_sensitive_traceback_and_logger_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    leaking_error = RuntimeError(f"{CONTENT} {SECRET} vector=[0.1, 0.2, 0.3]")
    service, _, _, _, _ = _service(
        events,
        embedding=_EmbeddingService(events, error=leaking_error),
    )
    captured_error: RagIngestionError | None = None

    with caplog.at_level(logging.ERROR, logger="rag-ingestion-test"):
        try:
            await service.ingest_sync(_request())
        except RagIngestionError as error:
            captured_error = error
            formatted = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            logging.getLogger("rag-ingestion-test").exception("safe ingestion failure")
        else:  # pragma: no cover - explicit assertion aid
            pytest.fail("RagIngestionError was not raised")

    assert captured_error is not None
    assert captured_error.__cause__ is None
    assert captured_error.__suppress_context__ is True
    visible = formatted + caplog.text
    for sensitive in (CONTENT, SECRET, "[0.1, 0.2, 0.3]"):
        assert sensitive not in visible


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["embed", "partial_upsert"])
async def test_cancellation_marks_failed_and_is_not_swallowed(stage: str) -> None:
    events: list[str] = []
    if stage == "embed":
        embedding = _EmbeddingService(events, error=asyncio.CancelledError())
        store = _ChunkStore(events)
    else:
        embedding = _EmbeddingService(events)
        store = _ChunkStore(events, cancel_after_write=True)
    service, documents, _, _, store = _service(
        events,
        embedding=embedding,
        store=store,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.ingest_sync(_request())

    assert events[-1] == "failed"
    assert len(documents.failed) == 1
    expected_code = "embedding_failed" if stage == "embed" else "chunk_store_failed"
    assert documents.failed[0]["failure_code"] == expected_code
    if stage == "embed":
        assert store.calls == []
    else:
        assert store.persisted_ids == {f"{DOCUMENT_ID}:000000"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_vector",
    [
        (0.1, 0.2),
        (0.1, True, 0.3),
        (0.1, "secret-vector", 0.3),
        (0.1, float("nan"), 0.3),
        (0.1, float("inf"), 0.3),
    ],
    ids=["dimension", "bool", "non-numeric", "nan", "infinity"],
)
async def test_invalid_embedding_vector_fails_before_store(bad_vector: tuple[Any, ...]) -> None:
    events: list[str] = []
    batch = RagEmbeddingBatch(
        vectors=(bad_vector,),
        provider="test-provider",
        model="test-model",
        version="test-v1",
        dimension=3,
    )
    service, documents, _, _, store = _service(
        events,
        embedding=_EmbeddingService(events, batch=batch),
    )

    with pytest.raises(RagIngestionError) as raised:
        await service.ingest_sync(_request())

    assert raised.value.code == "embedding_failed"
    assert store.calls == []
    assert documents.failed[0]["failure_code"] == "embedding_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "repository_error", "expected_stage", "expected_code"),
    [
        (
            "knowledge_base",
            RagPersistenceError("knowledge_base_store_error", SECRET),
            "knowledge_base_store",
            "knowledge_base_store_error",
        ),
        (
            "claim",
            RagKnowledgeBaseNotActiveError(),
            "knowledge_base",
            "knowledge_base_not_active",
        ),
        (
            "claim",
            RagDocumentArchivedError(),
            "archived",
            "document_archived",
        ),
        (
            "claim",
            RagPersistenceError("document_store_error", SECRET),
            "persistence",
            "rag_persistence_error",
        ),
    ],
)
async def test_c1_domain_errors_have_safe_precise_mapping(
    boundary: str,
    repository_error: RagPersistenceError,
    expected_stage: str,
    expected_code: str,
) -> None:
    events: list[str] = []
    knowledge_bases = _KnowledgeBaseRepository(events)
    documents = _DocumentRepository(events)
    if boundary == "knowledge_base":
        knowledge_bases.error = repository_error
    else:
        documents.claim_error = repository_error
    service, _, _, _, _ = _service(
        events,
        knowledge_bases=knowledge_bases,
        documents=documents,
    )

    with pytest.raises(RagIngestionError) as raised:
        await service.ingest_sync(_request())

    assert raised.value.stage == expected_stage
    assert raised.value.code == expected_code
    assert SECRET not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
