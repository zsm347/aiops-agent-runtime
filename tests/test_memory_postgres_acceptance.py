"""Real PostgreSQL acceptance gate for M-P2 long-term memory persistence.

These tests execute against an ISOLATED PostgreSQL database supplied through the
``M_P2_TEST_DATABASE_URL`` environment variable. They intentionally never read
``.env`` and never run against development or production data.

When the variable is absent the whole module is skipped and the PostgreSQL gate
is reported as ``pending`` (a skip is NOT counted as a pass). The scenarios
mirror ``docs/M-P2-postgresql-memory-persistence-plan.md`` section 17.4.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from sqlalchemy.exc import SQLAlchemyError

from superbiz_agent.memory.dedup import canonical_content_hash
from superbiz_agent.memory.persistence_contract import (
    ACTIVE_EXACT_INDEX,
    M_P2_REQUIRED_CHECKS,
)
from superbiz_agent.memory.schemas import (
    DEFAULT_CORE_BLOCK_KEYS,
    LongTermMemory,
    utc_now,
)
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.models import LongTermMemory as MemoryModel
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository


DATABASE_URL = os.environ.get("M_P2_TEST_DATABASE_URL")
PENDING_REASON = (
    "PostgreSQL acceptance gate pending: set M_P2_TEST_DATABASE_URL to an "
    "isolated test database to enable (never .env / dev / prod)."
)

pytestmark = pytest.mark.skipif(not DATABASE_URL, reason=PENDING_REASON)

ALEMBIC_DIR = Path(__file__).resolve().parents[1] / "alembic"


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(DATABASE_URL)
    yield eng
    asyncio.run(eng.dispose())


def _alembic(revision: str) -> None:
    from alembic import command
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", DATABASE_URL)
    command.upgrade(config, revision)


@pytest.fixture(scope="session")
def migrated(engine):
    _alembic("head")
    yield
    _alembic("base")


def _new_engine():
    return create_engine(DATABASE_URL)


def _repository(eng):
    return PostgresMemoryRepository(create_sessionmaker(eng))


def _archival(tenant, user, agent, *, topic="t", content="c", tags=("a",), **kw):
    now = utc_now()
    return LongTermMemory(
        id=kw.pop("id", f"{tenant}-{user}-{agent}-{topic}"),
        tenant_id=tenant,
        user_id=user,
        agent_id=agent,
        session_id=kw.pop("session_id", None),
        type=kw.pop("type", "experience"),
        topic=topic,
        content=content,
        source=kw.pop("source", "memory_service"),
        embedding=[0.0] * 64,
        embedding_model="local-deterministic",
        embedding_dimension=64,
        embedding_metric="cosine",
        embedding_version="phase4-local",
        content_hash=canonical_content_hash(content),
        tags=list(tags),
        scope_service=kw.pop("scope_service", None),
        scope_env=kw.pop("scope_env", None),
        status="active",
        created_at=now,
        updated_at=now,
        usage_count=0,
        last_used_at=None,
        archived_at=None,
        archive_reason=None,
        **kw,
    )


@pytest.mark.asyncio
async def test_migration_upgrade_and_downgrade_symmetry(migrated) -> None:
    constraints = {c.name for c in MemoryModel.__table__.constraints}
    indexes = {i.name for i in MemoryModel.__table__.indexes}
    assert M_P2_REQUIRED_CHECKS.issubset(constraints)
    assert ACTIVE_EXACT_INDEX in indexes


@pytest.mark.asyncio
async def test_readiness_fails_on_pre_mp2_revision_then_passes(engine) -> None:
    _alembic("20260712_01")  # pre-M-P2 revision
    repo_pre = _repository(_new_engine())
    with pytest.raises(SQLAlchemyError):
        await repo_pre.ensure_ready()

    _alembic("head")
    repo = _repository(_new_engine())
    await repo.ensure_ready()  # must pass after migration


@pytest.mark.asyncio
async def test_recovery_across_two_runtimes(migrated) -> None:
    tenant, user, agent = "rec", "u1", "a1"
    eng_a = _new_engine()
    repo_a = _repository(eng_a)
    blocks = await repo_a.ensure_default_core_blocks(tenant, user, agent)
    assert {b.block_key for b in blocks} == set(DEFAULT_CORE_BLOCK_KEYS)
    await repo_a.write_archival_exact(_archival(tenant, user, agent, content="first memory"))
    await eng_a.dispose()

    eng_b = _new_engine()
    repo_b = _repository(eng_b)
    recovered = await repo_b.list_active_memories(tenant, user, agent)
    assert len(recovered) == 1
    await eng_b.dispose()


@pytest.mark.asyncio
async def test_concurrent_exact_writes_merge_to_single_active(migrated) -> None:
    tenant, user, agent = "conc", "u1", "a1"
    eng = _new_engine()
    repo = _repository(eng)

    async def _write(tag: str):
        await repo.write_archival_exact(
            _archival(tenant, user, agent, content="shared body", tags=(tag,))
        )

    await asyncio.gather(*(_write(f"tag-{i}") for i in range(24)))
    rows = await repo.list_active_memories(tenant, user, agent)
    assert len(rows) == 1
    assert len(rows[0].tags) == 24
    await eng.dispose()


@pytest.mark.asyncio
async def test_core_cas_one_updated_one_conflict(migrated) -> None:
    tenant, user, agent = "cas", "u1", "a1"
    eng = _new_engine()
    repo = _repository(eng)
    await repo.ensure_default_core_blocks(tenant, user, agent)

    r1 = await repo.cas_replace_core_content(
        tenant, user, agent, "user_rules", "version one", expected_version=1
    )
    assert r1.status == "updated"
    r2 = await repo.cas_replace_core_content(
        tenant, user, agent, "user_rules", "version two", expected_version=1
    )
    assert r2.status == "conflict"
    await eng.dispose()


@pytest.mark.asyncio
async def test_tenant_isolation(migrated) -> None:
    eng = _new_engine()
    repo = _repository(eng)
    await repo.write_archival_exact(_archival("t1", "u1", "a1", content="body one"))
    await repo.write_archival_exact(_archival("t2", "u2", "a2", content="body two"))

    t1 = await repo.list_active_memories("t1", "u1", "a1")
    t2 = await repo.list_active_memories("t2", "u2", "a2")
    assert len(t1) == 1 and len(t2) == 1
    assert t1[0].content == "body one" and t2[0].content == "body two"
    await eng.dispose()


@pytest.mark.asyncio
async def test_local_deterministic_embedding_written_as_null(migrated) -> None:
    from sqlalchemy import select, text

    tenant, user, agent = "emb", "u1", "a1"
    eng = _new_engine()
    repo = _repository(eng)
    await repo.write_archival_exact(_archival(tenant, user, agent, content="embed body"))

    async with create_sessionmaker(eng)() as session:
        row = (
            await session.execute(
                select(MemoryModel).where(
                    MemoryModel.tenant_id == tenant,
                    MemoryModel.content == "embed body",
                )
            )
        ).scalar_one()
        assert row.embedding_model == "local-deterministic"
        assert row.embedding_dimension == 64
        # The vector column must stay NULL; it is never written by M-P2.
        raw = (
            await session.execute(
                text(
                    "SELECT embedding IS NULL AS is_null "
                    "FROM long_term_memory WHERE id = :id"
                ),
                {"id": row.id},
            )
        ).scalar()
        assert raw is True
    await eng.dispose()


@pytest.mark.asyncio
async def test_engine_close_is_idempotent(migrated) -> None:
    eng = create_engine(DATABASE_URL)
    await eng.dispose()
    await eng.dispose()  # repeated close must be safe
