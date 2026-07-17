from __future__ import annotations

from datetime import datetime, timezone

import pytest

from superbiz_agent.memory.backfill import MemoryEmbeddingBackfillService
from superbiz_agent.memory.embedding import (
    MemoryEmbeddingBatch,
    MemoryEmbeddingIdentity,
    MemoryQueryEmbedding,
)
from superbiz_agent.memory.errors import (
    MemoryEmbeddingUnavailableError,
    MemoryStoreContractError,
)
from superbiz_agent.memory.ports import EmbeddingBackfillCandidate


IDENTITY = MemoryEmbeddingIdentity(
    provider="dashscope-openai-compatible",
    model="text-embedding-v4",
    version="text-embedding-v4",
    dimension=1024,
)
NOW = datetime(2026, 7, 17, tzinfo=timezone.utc)


def _candidate(value: str) -> EmbeddingBackfillCandidate:
    return EmbeddingBackfillCandidate(
        id=value,
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        content=f"content-{value}",
        content_hash=value.ljust(64, "0"),
        updated_at=NOW,
    )


class _EmbeddingService:
    identity = IDENTITY

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[tuple[str, ...]] = []

    async def embed_documents(self, texts):
        values = tuple(texts)
        self.requests.append(values)
        if self.error is not None:
            raise self.error
        return MemoryEmbeddingBatch(
            vectors=tuple(tuple([float(index + 1)] * 1024) for index, _ in enumerate(values)),
            identity=self.identity,
        )

    async def embed_query(self, text: str) -> MemoryQueryEmbedding:
        raise AssertionError("backfill must not embed queries")

    async def aclose(self) -> None:
        return None


class _Repository:
    def __init__(self, candidates, *, apply_results=()) -> None:
        self.candidates = list(candidates)
        self.apply_results = list(apply_results)
        self.list_calls: list[tuple[str | None, int]] = []
        self.applied = []

    async def count_embedding_backfill_candidates(self, identity) -> int:
        assert identity == IDENTITY
        return len(self.candidates)

    async def list_embedding_backfill_candidates(self, identity, *, after_id, limit):
        assert identity == IDENTITY
        self.list_calls.append((after_id, limit))
        values = [candidate for candidate in self.candidates if after_id is None or candidate.id > after_id]
        return values[:limit]

    async def apply_embedding_backfill(self, candidate, embedding, identity):
        assert identity == IDENTITY
        self.applied.append((candidate.id, tuple(embedding)))
        return self.apply_results.pop(0) if self.apply_results else True


@pytest.mark.asyncio
async def test_backfill_batches_keyset_and_reports_concurrent_cas_skip() -> None:
    repository = _Repository(
        [_candidate("a"), _candidate("b"), _candidate("c")],
        apply_results=[True, False, True],
    )
    embedding = _EmbeddingService()

    report = await MemoryEmbeddingBackfillService(
        repository,
        embedding,
        batch_size=2,
    ).run()

    assert report.status == "passed"
    assert (report.eligible, report.scanned, report.embedded) == (3, 3, 3)
    assert (report.updated, report.concurrent_skipped, report.failed, report.batches) == (2, 1, 0, 2)
    assert repository.list_calls == [(None, 2), ("b", 2), ("c", 2)]
    assert embedding.requests == [("content-a", "content-b"), ("content-c",)]


@pytest.mark.asyncio
async def test_backfill_dry_run_never_calls_provider_or_writes() -> None:
    repository = _Repository([_candidate("a"), _candidate("b")])
    embedding = _EmbeddingService(error=AssertionError("must not be called"))

    report = await MemoryEmbeddingBackfillService(
        repository,
        embedding,
        batch_size=1,
    ).run(dry_run=True)

    assert report.status == "passed"
    assert (report.eligible, report.scanned, report.embedded, report.updated) == (2, 2, 0, 0)
    assert report.batches == 2
    assert embedding.requests == []
    assert repository.applied == []


@pytest.mark.asyncio
async def test_backfill_stops_with_stable_error_code_without_sensitive_values() -> None:
    repository = _Repository([_candidate("secret-id")])
    embedding = _EmbeddingService(error=MemoryEmbeddingUnavailableError())

    report = await MemoryEmbeddingBackfillService(
        repository,
        embedding,
        batch_size=10,
    ).run()

    assert report.status == "failed"
    assert report.failed == 1
    assert report.error_codes == ("memory_embedding_unavailable",)
    assert "secret-id" not in repr(report)
    assert "content" not in repr(report)


def test_backfill_rejects_deterministic_or_invalid_batch_configuration() -> None:
    embedding = _EmbeddingService()
    embedding.identity = MemoryEmbeddingIdentity(
        provider="local-deterministic",
        model="local-deterministic",
        version="phase4-local",
        dimension=64,
    )
    with pytest.raises(MemoryStoreContractError):
        MemoryEmbeddingBackfillService(_Repository([]), embedding, batch_size=10)
    with pytest.raises(MemoryStoreContractError):
        MemoryEmbeddingBackfillService(_Repository([]), _EmbeddingService(), batch_size=0)
