from __future__ import annotations

import os
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, text, update
from sqlalchemy.engine import make_url

from superbiz_agent.memory.backfill import MemoryEmbeddingBackfillService
from superbiz_agent.memory.dedup import canonical_content_hash
from superbiz_agent.memory.embedding import (
    MemoryEmbeddingBatch,
    MemoryEmbeddingIdentity,
    MemoryQueryEmbedding,
)
from superbiz_agent.memory.schemas import LongTermMemory, utc_now
from superbiz_agent.memory.search import MemorySearchService
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.models import LongTermMemory as MemoryModel
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository


DATABASE_URL = os.environ.get("M_P1_TEST_DATABASE_URL")
CONFIRMATION = os.environ.get("M_P1_TEST_DATABASE_DESTRUCTIVE_CONFIRM")
CONFIRMATION_VALUE = "ERASE_M_P1_ISOLATED_TEST_DATABASE"
DEDICATED_MARKERS = frozenset(
    {
        "superbiz-agent:m-p2-destructive-test-database:v1",
        "superbiz-agent:m-p1-retrieval-test-database:v1",
    }
)
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="M-P1 PostgreSQL acceptance pending: explicit isolated URL is missing.",
)

REAL_IDENTITY = MemoryEmbeddingIdentity(
    provider="dashscope-openai-compatible",
    model="text-embedding-v4",
    version="text-embedding-v4",
    dimension=1024,
)


def _vector(first: float, second: float = 0.0) -> list[float]:
    return [first, second] + [0.0] * 1022


class _StaticEmbeddingService:
    identity = REAL_IDENTITY

    async def embed_documents(self, texts):
        values = tuple(texts)
        return MemoryEmbeddingBatch(
            vectors=tuple(tuple(_vector(1.0, float(index) / 10)) for index, _ in enumerate(values)),
            identity=self.identity,
        )

    async def embed_query(self, text: str) -> MemoryQueryEmbedding:
        return MemoryQueryEmbedding(vector=tuple(_vector(1.0)), identity=self.identity)

    async def aclose(self) -> None:
        return None


def _database_url() -> str:
    assert DATABASE_URL is not None
    parsed = make_url(DATABASE_URL)
    database = (parsed.database or "").strip()
    if (
        CONFIRMATION != CONFIRMATION_VALUE
        or parsed.get_backend_name() != "postgresql"
        or parsed.get_driver_name() != "asyncpg"
        or not parsed.host
        or not database
        or database.lower() in {
            "postgres",
            "template0",
            "template1",
            "super_biz_agent",
            "superbiz_agent",
        }
        or ("m_p1_test" not in database.lower() and "m_p2_test" not in database.lower())
    ):
        raise RuntimeError("M-P1 acceptance database is not dedicated-test-only.")
    return DATABASE_URL


@pytest_asyncio.fixture
async def clean_database():
    engine = create_engine(_database_url())
    async with engine.connect() as connection:
        marker = await connection.scalar(
            text(
                "SELECT shobj_description(oid, 'pg_database') FROM pg_database "
                "WHERE datname = current_database()"
            )
        )
    if marker not in DEDICATED_MARKERS:
        await engine.dispose()
        raise RuntimeError("M-P1 acceptance database marker is invalid.")
    async with engine.begin() as connection:
        await connection.execute(delete(MemoryModel))
    repository = PostgresMemoryRepository(create_sessionmaker(engine))
    await repository.ensure_ready()
    try:
        yield engine, repository
    finally:
        async with engine.begin() as connection:
            await connection.execute(delete(MemoryModel))
        await engine.dispose()


def _memory(
    content: str,
    *,
    memory_id: str | None = None,
    tenant_id: str = "mpr1-tenant",
    user_id: str = "mpr1-user",
    agent_id: str = "mpr1-agent",
    memory_type: str = "experience",
    vector: list[float] | None = None,
    provider: str = REAL_IDENTITY.provider,
    model: str = REAL_IDENTITY.model,
    version: str = REAL_IDENTITY.version,
    dimension: int = 1024,
    scope_service: str | None = "orders-service",
    scope_env: str | None = "production",
    tags: list[str] | None = None,
) -> LongTermMemory:
    return LongTermMemory(
        id=memory_id or str(uuid4()),
        tenant_id=tenant_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id="mpr1-session",
        type=memory_type,  # type: ignore[arg-type]
        topic="mpr1-topic",
        content=content,
        embedding=vector if vector is not None else _vector(1.0),
        content_hash=canonical_content_hash(content),
        source="manual",
        embedding_provider=provider,
        embedding_model=model,
        embedding_version=version,
        embedding_dimension=dimension,
        tags=tags or ["database"],
        scope_service=scope_service,
        scope_env=scope_env,
    )


@pytest.mark.asyncio
async def test_real_vector_write_and_cross_engine_restart_recovery(clean_database) -> None:
    engine, repository = clean_database
    written = await repository.write_archival_exact(
        _memory("orders pool restart recovery", memory_id="mpr1-restart")
    )
    async with engine.connect() as connection:
        row = (
            await connection.execute(
                text(
                    "SELECT embedding IS NOT NULL AS vector_present, embedding_provider "
                    "FROM long_term_memory WHERE id = 'mpr1-restart'"
                )
            )
        ).mappings().one()
    assert row["vector_present"] is True
    assert row["embedding_provider"] == REAL_IDENTITY.provider

    restarted_engine = create_engine(_database_url())
    try:
        restarted = PostgresMemoryRepository(create_sessionmaker(restarted_engine))
        hits = await restarted.search_active_memories_by_vector(
            "mpr1-tenant",
            "mpr1-user",
            "mpr1-agent",
            _vector(1.0),
            REAL_IDENTITY,
            types=["experience"],
            min_similarity=0.99,
            limit=3,
        )
        assert [hit.memory.id for hit in hits] == [written.memory.id]
    finally:
        await restarted_engine.dispose()


@pytest.mark.asyncio
async def test_vector_query_enforces_filters_status_and_all_identity_axes(clean_database) -> None:
    engine, repository = clean_database
    candidates = [
        _memory("target vector", memory_id="mpr1-target", tags=["database"]),
        _memory("wrong service", memory_id="mpr1-service", scope_service="payments-service"),
        _memory("wrong env", memory_id="mpr1-env", scope_env="staging"),
        _memory("wrong tag", memory_id="mpr1-tag", tags=["cache"]),
        _memory("wrong tenant", memory_id="mpr1-tenant-x", tenant_id="other-tenant"),
        _memory("wrong user", memory_id="mpr1-user-x", user_id="other-user"),
        _memory("wrong agent", memory_id="mpr1-agent-x", agent_id="other-agent"),
        _memory("archived", memory_id="mpr1-archived"),
        _memory(
            "stale model",
            memory_id="mpr1-stale",
            provider="openai-compatible",
            model="text-embedding-3-large",
            version="text-embedding-3-large",
        ),
    ]
    for candidate in candidates:
        await repository.write_archival_exact(candidate)
    async with engine.begin() as connection:
        await connection.execute(
            update(MemoryModel)
            .where(MemoryModel.id == "mpr1-archived")
            .values(status="archived", archived_at=utc_now(), archive_reason="test")
        )

    hits = await repository.search_active_memories_by_vector(
        "mpr1-tenant",
        "mpr1-user",
        "mpr1-agent",
        _vector(1.0),
        REAL_IDENTITY,
        types=["experience"],
        scope_service="orders-service",
        scope_env="production",
        tags=["database"],
        min_similarity=0.99,
        limit=3,
    )

    assert [hit.memory.id for hit in hits] == ["mpr1-target"]


@pytest.mark.asyncio
async def test_search_service_uses_pgvector_repository_and_marks_returned(clean_database) -> None:
    _engine, repository = clean_database
    await repository.write_archival_exact(_memory("semantic orders connection exhaustion"))
    service = MemorySearchService(repository, _StaticEmbeddingService(), top_k=3, min_similarity=0.9)

    results = await service.search_memory(
        "mpr1-tenant",
        "mpr1-user",
        "mpr1-agent",
        "database pool unavailable",
        scope_service="orders-service",
        scope_env="production",
        tags=["database"],
    )

    assert len(results) == 1
    persisted = await repository.list_active_memories("mpr1-tenant", "mpr1-user", "mpr1-agent")
    assert persisted[0].usage_count == 1


@pytest.mark.asyncio
async def test_backfill_dry_run_apply_idempotence_and_vector_identity(clean_database) -> None:
    engine, repository = clean_database
    await repository.write_archival_exact(
        _memory(
            "legacy missing vector",
            memory_id="mpr1-backfill",
            vector=[0.0] * 64,
            provider="local-deterministic",
            model="local-deterministic",
            version="phase4-local",
            dimension=64,
        )
    )
    service = MemoryEmbeddingBackfillService(repository, _StaticEmbeddingService(), batch_size=2)

    dry_run = await service.run(dry_run=True)
    applied = await service.run()
    repeated = await service.run()

    assert (dry_run.eligible, dry_run.scanned, dry_run.updated) == (1, 1, 0)
    assert (applied.eligible, applied.updated, applied.failed) == (1, 1, 0)
    assert (repeated.eligible, repeated.scanned, repeated.updated) == (0, 0, 0)
    async with engine.connect() as connection:
        row = (
            await connection.execute(
                text(
                    "SELECT embedding IS NOT NULL AS vector_present, embedding_provider, "
                    "embedding_dimension FROM long_term_memory WHERE id = 'mpr1-backfill'"
                )
            )
        ).mappings().one()
    assert dict(row) == {
        "vector_present": True,
        "embedding_provider": REAL_IDENTITY.provider,
        "embedding_dimension": 1024,
    }


@pytest.mark.asyncio
async def test_backfill_cas_does_not_overwrite_concurrent_content_change(clean_database) -> None:
    engine, repository = clean_database
    await repository.write_archival_exact(
        _memory(
            "content before concurrent edit",
            memory_id="mpr1-cas",
            vector=[0.0] * 64,
            provider="local-deterministic",
            model="local-deterministic",
            version="phase4-local",
            dimension=64,
        )
    )
    candidate = (
        await repository.list_embedding_backfill_candidates(
            REAL_IDENTITY,
            after_id=None,
            limit=10,
        )
    )[0]
    changed_content = "content after concurrent edit"
    async with engine.begin() as connection:
        await connection.execute(
            update(MemoryModel)
            .where(MemoryModel.id == candidate.id)
            .values(
                content=changed_content,
                content_hash=canonical_content_hash(changed_content),
                updated_at=MemoryModel.updated_at + text("interval '1 second'"),
            )
        )

    applied = await repository.apply_embedding_backfill(
        candidate,
        _vector(1.0),
        REAL_IDENTITY,
    )

    assert applied is False
    async with engine.connect() as connection:
        row = (
            await connection.execute(
                text(
                    "SELECT content, embedding IS NULL AS vector_missing "
                    "FROM long_term_memory WHERE id = 'mpr1-cas'"
                )
            )
        ).mappings().one()
    assert row["content"] == changed_content
    assert row["vector_missing"] is True


@pytest.mark.asyncio
async def test_exact_dedupe_remains_hash_only_and_merges_tags(clean_database) -> None:
    _engine, repository = clean_database
    first = await repository.write_archival_exact(
        _memory("same canonical archival content", memory_id="mpr1-exact-a", tags=["first"])
    )
    duplicate = await repository.write_archival_exact(
        _memory(
            "same canonical archival content!!!",
            memory_id="mpr1-exact-b",
            vector=_vector(0.0, 1.0),
            tags=["second"],
        )
    )

    assert first.status == "written"
    assert duplicate.status == "duplicate_skipped"
    assert duplicate.memory.id == first.memory.id
    assert duplicate.memory.tags == ["first", "second"]


@pytest.mark.asyncio
async def test_backfill_only_selects_active_archival_missing_or_stale_rows(clean_database) -> None:
    engine, repository = clean_database
    current = _memory("current vector", memory_id="mpr1-current")
    archived = _memory(
        "archived missing vector",
        memory_id="mpr1-old",
        vector=[0.0] * 64,
        provider="local-deterministic",
        model="local-deterministic",
        version="phase4-local",
        dimension=64,
    )
    await repository.write_archival_exact(current)
    await repository.write_archival_exact(archived)
    async with engine.begin() as connection:
        await connection.execute(
            update(MemoryModel)
            .where(MemoryModel.id == archived.id)
            .values(status="archived", archived_at=utc_now(), archive_reason="test")
        )

    assert await repository.count_embedding_backfill_candidates(REAL_IDENTITY) == 0
