"""M-P2 acceptance contracts against one explicitly supplied isolated PostgreSQL DB.

The module reads only the explicit URL and independent destructive-confirmation
gate variables. It never constructs Settings without ``_env_file=None``. Without
the URL every test is skipped and the PostgreSQL gate remains pending; skips are
not acceptance passes.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from alembic import command
from alembic.config import Config
import pytest
import pytest_asyncio
from sqlalchemy import event, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from superbiz_agent.config import Settings
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore
from superbiz_agent.memory.dedup import canonical_content_hash
from superbiz_agent.memory.errors import (
    CoreMemoryContractError,
    MemoryExactConflictUnresolvedError,
    MemoryStoreContractError,
    MemoryStoreUnavailableError,
)
from superbiz_agent.memory.persistence_contract import (
    ACTIVE_CONTENT_HASH_CHECK,
    ACTIVE_EXACT_INDEX,
    CORE_UNIQUE_CONSTRAINT,
    M_P2_REQUIRED_CHECKS,
    SCOPE_ENV_NONBLANK_CHECK,
)
from superbiz_agent.memory.runtime import build_memory_runtime
from superbiz_agent.memory.schemas import DEFAULT_CORE_BLOCK_KEYS, LongTermMemory, utc_now
from superbiz_agent.persistence.alembic_url import (
    CONNECTION_VALIDATOR_ATTRIBUTE,
    EXPLICIT_DATABASE_URL_ATTRIBUTE,
)
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.models import (
    AgentCoreMemoryBlock,
    LongTermMemory as MemoryModel,
)
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository
from superbiz_agent.tools.builtin import build_builtin_tools


DATABASE_URL = os.environ.get("M_P2_TEST_DATABASE_URL")
DESTRUCTIVE_CONFIRMATION = os.environ.get("M_P2_TEST_DATABASE_DESTRUCTIVE_CONFIRM")
PENDING_REASON = (
    "PostgreSQL acceptance pending: M_P2_TEST_DATABASE_URL is not set to an "
    "explicit isolated test database."
)
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason=PENDING_REASON)

ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_DIR = ROOT / "alembic"
PRE_M_P2_REVISION = "20260712_01"
HEAD_REVISION = "20260717_01"
DESTRUCTIVE_CONFIRMATION_VALUE = "ERASE_M_P2_ISOLATED_TEST_DATABASE"
DEDICATED_DATABASE_MARKER = "superbiz-agent:m-p2-destructive-test-database:v1"
_DEFAULT_OR_PRODUCTION_DATABASE_NAMES = frozenset(
    {"postgres", "template0", "template1", "super_biz_agent", "superbiz_agent"}
)
_APPLICATION_TABLES = (
    "agent_rollout_event",
    "long_term_memory",
    "agent_core_memory_block",
    "rag_knowledge_base",
    "rag_document",
)
_RAW_DRIVER_OUTPUT_MARKERS = (
    "asyncpg",
    "sqlalchemy.exc",
    "traceback (most recent call last)",
    "detail:",
)
_DATABASE_MARKER_QUERY = text(
    "SELECT shobj_description(oid, 'pg_database') FROM pg_database "
    "WHERE datname = current_database()"
)


class _AsyncStartBarrier:
    def __init__(self, parties: int) -> None:
        self._parties = parties
        self._arrived = 0
        self._lock = asyncio.Lock()
        self._all_ready = asyncio.Event()
        self._start = asyncio.Event()

    async def arrive_and_wait(self) -> None:
        async with self._lock:
            self._arrived += 1
            if self._arrived == self._parties:
                self._all_ready.set()
        await self._start.wait()

    async def wait_until_ready(self) -> None:
        await asyncio.wait_for(self._all_ready.wait(), timeout=10)

    def release(self) -> None:
        self._start.set()


def _assert_no_sensitive_or_driver_output(
    *surfaces: str,
    secrets: tuple[str, ...] = (),
) -> None:
    combined = "\n".join(surfaces)
    if any(secret and secret in combined for secret in secrets):
        pytest.fail("PostgreSQL acceptance output exposed protected input.")
    lowered = combined.lower()
    if any(marker in lowered for marker in _RAW_DRIVER_OUTPUT_MARKERS):
        pytest.fail("PostgreSQL acceptance output exposed raw driver diagnostics.")


async def _raw_memory_rows(engine, *, tenant_id: str) -> list[dict[str, Any]]:
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT * FROM long_term_memory "
                    "WHERE tenant_id=:tenant_id ORDER BY id"
                ),
                {"tenant_id": tenant_id},
            )
        ).mappings().all()
    return [dict(row) for row in rows]


async def _raw_core_rows(engine, *, tenant_id: str) -> list[dict[str, Any]]:
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT * FROM agent_core_memory_block "
                    "WHERE tenant_id=:tenant_id ORDER BY id"
                ),
                {"tenant_id": tenant_id},
            )
        ).mappings().all()
    return [dict(row) for row in rows]


def _database_url() -> str:
    assert DATABASE_URL is not None
    return DATABASE_URL


def _database_url_secrets() -> tuple[str, ...]:
    database_url = _database_url()
    try:
        parsed = make_url(database_url)
    except (ArgumentError, TypeError, ValueError):
        return (database_url,)
    values = [
        database_url,
        parsed.username,
        str(parsed.password) if parsed.password is not None else None,
        parsed.host,
        parsed.database,
    ]
    secrets: list[str] = []
    for value in values:
        if not value:
            continue
        for candidate in (value, quote(value, safe="")):
            if candidate and candidate not in secrets:
                secrets.append(candidate)
    return tuple(secrets)


def _validated_destructive_database_name(
    database_url: str,
    confirmation: str | None,
) -> str:
    if confirmation != DESTRUCTIVE_CONFIRMATION_VALUE:
        raise RuntimeError("PostgreSQL destructive test confirmation is missing or invalid.")
    try:
        parsed = make_url(database_url)
    except (ArgumentError, TypeError, ValueError):
        raise RuntimeError("PostgreSQL destructive test URL is invalid.") from None
    database_name = (parsed.database or "").strip()
    if (
        parsed.get_backend_name() != "postgresql"
        or parsed.get_driver_name() != "asyncpg"
        or not parsed.host
        or not database_name
        or database_name.lower() in _DEFAULT_OR_PRODUCTION_DATABASE_NAMES
        or "m_p2_test" not in database_name.lower()
    ):
        raise RuntimeError("PostgreSQL destructive test URL is not dedicated-test-only.")
    return database_name


def _validate_dedicated_connection(connection, *, expected_database: str) -> None:
    actual_database = connection.scalar(text("SELECT current_database()"))
    marker = connection.scalar(_DATABASE_MARKER_QUERY)
    if actual_database != expected_database or marker != DEDICATED_DATABASE_MARKER:
        raise RuntimeError("PostgreSQL database is not marked as dedicated M-P2 test-only.")


async def _assert_dedicated_test_database(engine, *, expected_database: str) -> None:
    async with engine.connect() as connection:
        actual_database = await connection.scalar(text("SELECT current_database()"))
        marker = await connection.scalar(_DATABASE_MARKER_QUERY)
    if actual_database != expected_database or marker != DEDICATED_DATABASE_MARKER:
        raise RuntimeError("PostgreSQL database is not marked as dedicated M-P2 test-only.")


def _settings(**overrides: Any) -> Settings:
    values = {
        "database_url": _database_url(),
        "memory_enabled": True,
        "memory_store_backend": "postgres",
        "rollout_store_backend": "memory",
        "model_provider": "stub",
        "rag_enabled": False,
        "rag_fixture_mode": True,
        "context_content_compression_backend": "deterministic",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _alembic_config() -> Config:
    expected_database = _validated_destructive_database_name(
        _database_url(),
        DESTRUCTIVE_CONFIRMATION,
    )
    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    config.attributes[EXPLICIT_DATABASE_URL_ATTRIBUTE] = _database_url()

    def validate_connection(connection) -> None:
        _validate_dedicated_connection(
            connection,
            expected_database=expected_database,
        )

    config.attributes[CONNECTION_VALIDATOR_ATTRIBUTE] = validate_connection
    return config


async def _upgrade(revision: str = "head") -> None:
    _validated_destructive_database_name(_database_url(), DESTRUCTIVE_CONFIRMATION)
    await asyncio.to_thread(command.upgrade, _alembic_config(), revision)


async def _downgrade(revision: str) -> None:
    _validated_destructive_database_name(_database_url(), DESTRUCTIVE_CONFIRMATION)
    await asyncio.to_thread(command.downgrade, _alembic_config(), revision)


def _repository(engine) -> PostgresMemoryRepository:
    return PostgresMemoryRepository(create_sessionmaker(engine))


def _archival(
    tenant: str,
    user: str,
    agent: str,
    *,
    content: str,
    memory_id: str | None = None,
    topic: str = "topic",
    tags: tuple[str, ...] = ("tag",),
    dimension: int = 64,
    memory_type: str = "experience",
    scope_service: str | None = None,
    scope_env: str | None = None,
) -> LongTermMemory:
    now = utc_now()
    return LongTermMemory(
        id=memory_id or str(uuid4()),
        tenant_id=tenant,
        user_id=user,
        agent_id=agent,
        session_id="session",
        type=memory_type,  # type: ignore[arg-type]
        topic=topic,
        content=content,
        source="manual",
        embedding=[0.0] * dimension,
        embedding_model="local-deterministic",
        embedding_dimension=dimension,
        embedding_metric="cosine",
        embedding_version=f"local-{dimension}",
        content_hash=canonical_content_hash(content),
        tags=list(tags),
        scope_service=scope_service,
        scope_env=scope_env,
        status="active",
        created_at=now,
        updated_at=now,
    )


async def _truncate_memory(engine) -> None:
    async with engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE TABLE long_term_memory, agent_core_memory_block")
        )


async def _assert_database_has_no_application_data(engine) -> None:
    async with engine.connect() as connection:
        for table_name in _APPLICATION_TABLES:
            exists = await connection.scalar(
                text("SELECT to_regclass(:table_name) IS NOT NULL"),
                {"table_name": table_name},
            )
            if not exists:
                continue
            count = await connection.scalar(text(f"SELECT count(*) FROM {table_name}"))
            if int(count or 0):
                pytest.fail(
                    "Isolated PostgreSQL gate refused a database containing application data."
                )


async def _set_database_marker(engine, *, database_name: str, marker: str) -> None:
    quoted_database = engine.dialect.identifier_preparer.quote(database_name)
    escaped_marker = marker.replace("'", "''")
    async with engine.begin() as connection:
        await connection.execute(
            text(f"COMMENT ON DATABASE {quoted_database} IS '{escaped_marker}'")
        )


@pytest_asyncio.fixture(scope="session", autouse=True)
async def isolated_database_gate():
    expected_database = _validated_destructive_database_name(
        _database_url(),
        DESTRUCTIVE_CONFIRMATION,
    )
    engine = create_engine(_database_url())
    await _assert_dedicated_test_database(
        engine,
        expected_database=expected_database,
    )
    await _assert_database_has_no_application_data(engine)
    await engine.dispose()
    await _downgrade("base")
    await _upgrade("head")
    yield
    cleanup_engine = create_engine(_database_url())
    with suppress(Exception):
        await _truncate_memory(cleanup_engine)
    await cleanup_engine.dispose()
    await _downgrade("base")


@pytest_asyncio.fixture
async def clean_head(isolated_database_gate):
    await _upgrade("head")
    engine = create_engine(_database_url())
    await _truncate_memory(engine)
    yield engine
    await _truncate_memory(engine)
    await engine.dispose()


@pytest_asyncio.fixture
async def clean_pre_mp2(isolated_database_gate):
    await _downgrade(PRE_M_P2_REVISION)
    engine = create_engine(_database_url())
    await _truncate_memory(engine)
    yield engine
    await _truncate_memory(engine)
    await engine.dispose()
    await _upgrade("head")


@pytest.mark.asyncio
async def test_full_upgrade_and_downgrade_change_actual_database_state(
    isolated_database_gate,
) -> None:
    await _downgrade("base")
    engine = create_engine(_database_url())
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT to_regclass('long_term_memory')")) is None
    await engine.dispose()

    await _upgrade("head")
    engine = create_engine(_database_url())
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT to_regclass('long_term_memory')"))
        assert await connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            HEAD_REVISION
        )
    await engine.dispose()


@pytest.mark.asyncio
async def test_database_shared_marker_positive_and_negative_gate(clean_head) -> None:
    expected_database = _validated_destructive_database_name(
        _database_url(),
        DESTRUCTIVE_CONFIRMATION,
    )
    await _assert_dedicated_test_database(
        clean_head,
        expected_database=expected_database,
    )

    marker_changed = False
    try:
        await _set_database_marker(
            clean_head,
            database_name=expected_database,
            marker="invalid-m-p2-marker",
        )
        marker_changed = True
        with pytest.raises(RuntimeError, match="not marked"):
            await _assert_dedicated_test_database(
                clean_head,
                expected_database=expected_database,
            )
    finally:
        if marker_changed:
            await _set_database_marker(
                clean_head,
                database_name=expected_database,
                marker=DEDICATED_DATABASE_MARKER,
            )

    await _assert_dedicated_test_database(
        clean_head,
        expected_database=expected_database,
    )


@pytest.mark.asyncio
async def test_readiness_rejects_pre_mp2_and_passes_after_upgrade(clean_head) -> None:
    await _downgrade(PRE_M_P2_REVISION)
    pre_engine = create_engine(_database_url())
    with pytest.raises(MemoryStoreContractError):
        await _repository(pre_engine).ensure_ready()
    await pre_engine.dispose()

    await _upgrade("head")
    post_engine = create_engine(_database_url())
    await _repository(post_engine).ensure_ready()
    await post_engine.dispose()


@pytest.mark.asyncio
async def test_runtime_a_writes_closes_and_runtime_b_recovers(clean_head) -> None:
    trace_a = InMemoryRolloutEventStore()
    runtime_a = build_memory_runtime(_settings(), trace_store=trace_a)
    blocks = await runtime_a.repository.ensure_default_core_blocks("recover", "user", "agent")
    assert [block.block_key for block in blocks] == list(DEFAULT_CORE_BLOCK_KEYS)
    core_result = await runtime_a.repository.cas_replace_core_content(
        "recover",
        "user",
        "agent",
        "user_rules",
        "persisted core",
        expected_version=1,
    )
    assert core_result.status == "updated"
    await runtime_a.repository.write_archival_exact(
        _archival("recover", "user", "agent", content="persisted archival")
    )
    await runtime_a.aclose()

    runtime_b = build_memory_runtime(
        _settings(), trace_store=InMemoryRolloutEventStore()
    )
    recovered_core = await runtime_b.repository.list_active_core_blocks(
        "recover", "user", "agent"
    )
    recovered_archival = await runtime_b.repository.list_active_memories(
        "recover", "user", "agent"
    )
    assert next(block for block in recovered_core if block.block_key == "user_rules").content == (
        "persisted core"
    )
    assert [memory.content for memory in recovered_archival] == ["persisted archival"]
    await runtime_b.aclose()


@pytest.mark.asyncio
async def test_two_engines_initialize_exactly_three_core_rows(clean_head) -> None:
    engine_a = create_engine(_database_url())
    engine_b = create_engine(_database_url())
    barrier = _AsyncStartBarrier(2)

    async def initialize(repository: PostgresMemoryRepository):
        await barrier.arrive_and_wait()
        return await repository.ensure_default_core_blocks(
            "core-init", "user", "agent"
        )

    first_task = asyncio.create_task(initialize(_repository(engine_a)))
    second_task = asyncio.create_task(initialize(_repository(engine_b)))
    await barrier.wait_until_ready()
    barrier.release()
    first, second = await asyncio.gather(first_task, second_task)
    assert [block.block_key for block in first] == list(DEFAULT_CORE_BLOCK_KEYS)
    assert [block.block_key for block in second] == list(DEFAULT_CORE_BLOCK_KEYS)
    async with engine_a.connect() as connection:
        count = await connection.scalar(
            text(
                "SELECT count(*) FROM agent_core_memory_block "
                "WHERE tenant_id='core-init' AND user_id='user' AND agent_id='agent'"
            )
        )
    assert count == 3
    await asyncio.gather(engine_a.dispose(), engine_b.dispose())


@pytest.mark.asyncio
async def test_archived_default_rolls_back_missing_default_inserts(clean_head) -> None:
    engine = clean_head
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO agent_core_memory_block "
                "(id, tenant_id, user_id, agent_id, block_key, description, content, "
                "max_tokens, version, read_only, source, content_hash, status) "
                "VALUES (:id, 'core-rollback', 'user', 'agent', 'service_notes', "
                "'notes', '', 600, 1, false, 'memory_service', '', 'archived')"
            ),
            {"id": str(uuid4())},
        )
    with pytest.raises(CoreMemoryContractError):
        await _repository(engine).ensure_default_core_blocks(
            "core-rollback", "user", "agent"
        )
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT block_key, status FROM agent_core_memory_block "
                    "WHERE tenant_id='core-rollback' ORDER BY block_key"
                )
            )
        ).all()
    assert rows == [("service_notes", "archived")]


@pytest.mark.asyncio
async def test_twenty_four_concurrent_exact_writes_merge_all_tags(clean_head) -> None:
    repository = _repository(clean_head)
    barrier = _AsyncStartBarrier(24)

    async def write_one(index: int):
        await barrier.arrive_and_wait()
        return await repository.write_archival_exact(
            _archival(
                "exact-24",
                "user",
                "agent",
                memory_id=str(uuid4()),
                content="shared exact content",
                tags=(f"tag-{index}",),
            )
        )

    tasks = [asyncio.create_task(write_one(index)) for index in range(24)]
    await barrier.wait_until_ready()
    barrier.release()
    results = await asyncio.gather(*tasks)
    rows = await repository.list_active_memories("exact-24", "user", "agent")
    assert sum(result.status == "written" for result in results) == 1
    assert len(rows) == 1
    assert set(rows[0].tags) == {f"tag-{index}" for index in range(24)}


@pytest.mark.asyncio
async def test_exact_and_unrelated_primary_key_conflict_is_unresolved(clean_head) -> None:
    repository = _repository(clean_head)
    exact = _archival(
        "pk-conflict", "user", "agent", memory_id="exact-row", content="same"
    )
    unrelated = _archival(
        "pk-conflict", "user", "agent", memory_id="occupied-id", content="other"
    )
    await repository.write_archival_exact(exact)
    await repository.write_archival_exact(unrelated)
    before = await _raw_memory_rows(clean_head, tenant_id="pk-conflict")

    try:
        await repository.write_archival_exact(
            _archival(
                "pk-conflict",
                "user",
                "agent",
                memory_id="occupied-id",
                content="same",
            )
        )
    except MemoryExactConflictUnresolvedError as exc:
        conflict_message = str(exc)
    except Exception:
        raise AssertionError(
            "Primary-key/exact conflict escaped its safe contract error."
        ) from None
    else:
        pytest.fail("Primary-key/exact conflict unexpectedly succeeded.")
    _assert_no_sensitive_or_driver_output(
        conflict_message,
        secrets=("exact-row", "occupied-id", "same", "other"),
    )
    after = await _raw_memory_rows(clean_head, tenant_id="pk-conflict")
    if after != before:
        pytest.fail("Primary-key/exact conflict changed persisted rows.")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("targets", "expected_statuses"),
    [(("target-a", "target-b"), {"updated", "conflict"}),
     (("same-target", "same-target"), {"updated", "unchanged"})],
)
async def test_concurrent_core_cas_contracts(clean_head, targets, expected_statuses) -> None:
    engine_a = create_engine(_database_url())
    engine_b = create_engine(_database_url())
    repository_a = _repository(engine_a)
    repository_b = _repository(engine_b)
    await repository_a.ensure_default_core_blocks("cas", "user", "agent")
    barrier = _AsyncStartBarrier(2)

    async def replace(repository, target):
        await barrier.arrive_and_wait()
        return await repository.cas_replace_core_content(
            "cas", "user", "agent", "user_rules", target, expected_version=1
        )

    first_task = asyncio.create_task(replace(repository_a, targets[0]))
    second_task = asyncio.create_task(replace(repository_b, targets[1]))
    await barrier.wait_until_ready()
    barrier.release()
    first, second = await asyncio.gather(first_task, second_task)
    assert {first.status, second.status} == expected_statuses
    await asyncio.gather(engine_a.dispose(), engine_b.dispose())


@pytest.mark.asyncio
async def test_read_only_switch_rejects_old_cas(clean_head) -> None:
    admin_repository = _repository(clean_head)
    await admin_repository.ensure_default_core_blocks("readonly", "user", "agent")
    cas_engine = create_engine(_database_url())
    cas_repository = _repository(cas_engine)
    cas_started = asyncio.Event()

    @event.listens_for(cas_engine.sync_engine, "before_cursor_execute")
    def observe_cas(_conn, _cursor, statement, _params, _context, _many):
        if statement.startswith("UPDATE agent_core_memory_block"):
            cas_started.set()

    async with clean_head.begin() as connection:
        await connection.execute(
            text(
                "SELECT id FROM agent_core_memory_block "
                "WHERE tenant_id='readonly' AND user_id='user' AND agent_id='agent' "
                "AND block_key='user_rules' FOR UPDATE"
            )
        )
        cas_task = asyncio.create_task(
            cas_repository.cas_replace_core_content(
                "readonly",
                "user",
                "agent",
                "user_rules",
                "old cas",
                expected_version=1,
            )
        )
        await asyncio.wait_for(cas_started.wait(), timeout=5)
        await connection.execute(
            update(AgentCoreMemoryBlock)
            .where(
                AgentCoreMemoryBlock.tenant_id == "readonly",
                AgentCoreMemoryBlock.block_key == "user_rules",
            )
            .values(read_only=True)
        )
    result = await cas_task
    assert result.status == "read_only"
    await cas_engine.dispose()


@pytest.mark.asyncio
async def test_full_identity_type_and_scope_isolation(clean_head) -> None:
    repository = _repository(clean_head)
    variants = [
        ("tenant-a", "user", "agent", "experience", None, None),
        ("tenant-b", "user", "agent", "experience", None, None),
        ("tenant-a", "other-user", "agent", "experience", None, None),
        ("tenant-a", "user", "other-agent", "experience", None, None),
        ("tenant-a", "user", "agent", "knowledge", None, None),
        ("tenant-a", "user", "agent", "experience", "svc", "prod"),
    ]
    written_ids: set[str] = set()
    for tenant, user, agent, memory_type, service, env in variants:
        result = await repository.write_archival_exact(
            _archival(
                tenant,
                user,
                agent,
                content="same isolated content",
                memory_type=memory_type,
                scope_service=service,
                scope_env=env,
            )
        )
        assert result.status == "written"
        written_ids.add(result.memory.id)

    expected_counts = {
        ("tenant-a", "user", "agent"): 3,
        ("tenant-b", "user", "agent"): 1,
        ("tenant-a", "other-user", "agent"): 1,
        ("tenant-a", "user", "other-agent"): 1,
    }
    observed_ids: set[str] = set()
    for identity, expected_count in expected_counts.items():
        rows = await repository.list_active_memories(*identity)
        assert len(rows) == expected_count
        assert all(
            (row.tenant_id, row.user_id, row.agent_id) == identity for row in rows
        )
        observed_ids.update(row.id for row in rows)
    assert observed_ids == written_ids

    knowledge = await repository.list_active_memories(
        "tenant-a", "user", "agent", types=["knowledge"]
    )
    scoped = await repository.list_active_memories(
        "tenant-a",
        "user",
        "agent",
        scope_service="svc",
        scope_env="prod",
    )
    assert [(row.type, row.scope_service, row.scope_env) for row in knowledge] == [
        ("knowledge", None, None)
    ]
    assert [(row.type, row.scope_service, row.scope_env) for row in scoped] == [
        ("experience", "svc", "prod")
    ]


@pytest.mark.asyncio
async def test_archived_exact_row_does_not_block_new_active(clean_head) -> None:
    repository = _repository(clean_head)
    first = await repository.write_archival_exact(
        _archival("archived", "user", "agent", content="reusable content")
    )
    async with clean_head.begin() as connection:
        await connection.execute(
            update(MemoryModel)
            .where(MemoryModel.id == first.memory.id)
            .values(status="archived")
        )
    archived_before = (
        await _raw_memory_rows(clean_head, tenant_id="archived")
    )[0]
    second = await repository.write_archival_exact(
        _archival("archived", "user", "agent", content="reusable content")
    )
    assert second.status == "written"
    assert second.memory.id != first.memory.id
    rows = await _raw_memory_rows(clean_head, tenant_id="archived")
    archived_after = next(row for row in rows if row["id"] == first.memory.id)
    active_after = next(row for row in rows if row["id"] == second.memory.id)
    if archived_after != archived_before:
        pytest.fail("Writing a replacement changed the archived exact row.")
    assert active_after["status"] == "active"
    assert active_after["content_hash"] == archived_after["content_hash"]


@pytest.mark.asyncio
async def test_usage_count_has_no_lost_increment(clean_head) -> None:
    repository = _repository(clean_head)
    written = await repository.write_archival_exact(
        _archival("usage", "user", "agent", content="usage content")
    )
    barrier = _AsyncStartBarrier(24)

    async def increment_once() -> None:
        await barrier.arrive_and_wait()
        await repository.mark_returned(
            "usage", "user", "agent", [written.memory.id]
        )

    tasks = [asyncio.create_task(increment_once()) for _ in range(24)]
    await barrier.wait_until_ready()
    barrier.release()
    await asyncio.gather(*tasks)
    row = (await repository.list_active_memories("usage", "user", "agent"))[0]
    assert row.usage_count == 24


@pytest.mark.asyncio
async def test_local_deterministic_dimensions_write_null(clean_head) -> None:
    repository = _repository(clean_head)
    first = await repository.write_archival_exact(
        _archival("dimension", "user", "agent", content="dim-64", dimension=64)
    )
    second = await repository.write_archival_exact(
        _archival("dimension", "user", "agent", content="dim-17", dimension=17)
    )
    async with clean_head.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT id, embedding IS NULL AS embedding_is_null, "
                    "embedding_model, embedding_dimension, embedding_metric, "
                    "embedding_version "
                    "FROM long_term_memory WHERE id IN (:first, :second) ORDER BY id"
                ),
                {"first": first.memory.id, "second": second.memory.id},
            )
        ).mappings().all()
    by_id = {str(row["id"]): dict(row) for row in rows}
    assert by_id == {
        first.memory.id: {
            "id": first.memory.id,
            "embedding_is_null": True,
            "embedding_model": "local-deterministic",
            "embedding_dimension": 64,
            "embedding_metric": "cosine",
            "embedding_version": "local-64",
        },
        second.memory.id: {
            "id": second.memory.id,
            "embedding_is_null": True,
            "embedding_model": "local-deterministic",
            "embedding_dimension": 17,
            "embedding_metric": "cosine",
            "embedding_version": "local-17",
        },
    }


@pytest.mark.asyncio
async def test_existing_vector_survives_scalar_read_merge_and_core_write(clean_head) -> None:
    vector_text = "[" + ",".join("0" for _ in range(1024)) + "]"
    memory_id = str(uuid4())
    content = "vector preservation"
    created_at = utc_now()
    async with clean_head.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO long_term_memory "
                "(id, tenant_id, user_id, agent_id, session_id, type, topic, content, "
                "source, embedding, "
                "embedding_model, embedding_dimension, embedding_metric, embedding_version, "
                "created_at, updated_at, usage_count, last_used_at, tags, content_hash, status) "
                "VALUES (:id, 'vector', 'user', 'agent', 'original-session', 'experience', "
                "'original-topic', :content, 'rollout', "
                "CAST(:embedding AS vector), 'provider-model', 1024, 'cosine', 'v1', "
                ":created_at, :created_at, 7, :created_at, '[\"old\"]'::jsonb, :hash, "
                "'active')"
            ),
            {
                "id": memory_id,
                "content": content,
                "embedding": vector_text,
                "hash": canonical_content_hash(content),
                "created_at": created_at,
            },
        )
        before = dict(
            (
                await connection.execute(
                    text(
                        "SELECT id, tenant_id, user_id, agent_id, session_id, type, topic, "
                        "content, source, embedding::text AS embedding, embedding_model, "
                        "embedding_dimension, embedding_metric, embedding_version, created_at, "
                        "updated_at, usage_count, last_used_at, status, archived_at, "
                        "archive_reason, tags, scope_service, scope_env, content_hash "
                        "FROM long_term_memory WHERE id=:id"
                    ),
                    {"id": memory_id},
                )
            ).mappings().one()
        )
    repository = _repository(clean_head)
    await repository.list_active_memories("vector", "user", "agent")
    await repository.write_archival_exact(
        _archival("vector", "user", "agent", content=content, tags=("new",))
    )
    await repository.ensure_default_core_blocks("vector", "user", "agent")
    await repository.cas_replace_core_content(
        "vector", "user", "agent", "user_rules", "core", expected_version=1
    )
    async with clean_head.connect() as connection:
        after = dict(
            (
                await connection.execute(
                    text(
                        "SELECT id, tenant_id, user_id, agent_id, session_id, type, topic, "
                        "content, source, embedding::text AS embedding, embedding_model, "
                        "embedding_dimension, embedding_metric, embedding_version, created_at, "
                        "updated_at, usage_count, last_used_at, status, archived_at, "
                        "archive_reason, tags, scope_service, scope_env, content_hash "
                        "FROM long_term_memory WHERE id=:id"
                    ),
                    {"id": memory_id},
                )
            ).mappings().one()
        )
    before_without_merge_fields = {
        key: value for key, value in before.items() if key not in {"tags", "updated_at"}
    }
    after_without_merge_fields = {
        key: value for key, value in after.items() if key not in {"tags", "updated_at"}
    }
    if after_without_merge_fields != before_without_merge_fields:
        pytest.fail("Scalar/Core operations changed preserved vector-row metadata.")
    assert before["tags"] == ["old"]
    assert after["tags"] == ["old", "new"]
    assert after["updated_at"] >= before["updated_at"]


@pytest.mark.asyncio
async def test_active_null_and_malformed_hash_cannot_bypass_constraints(clean_head) -> None:
    statement = text(
        "INSERT INTO long_term_memory "
        "(id, tenant_id, user_id, agent_id, type, topic, content, source, "
        "embedding_model, embedding_dimension, embedding_metric, embedding_version, "
        "tags, content_hash, status) VALUES "
        "(:id, 'hash-negative', 'user', 'agent', 'experience', 'topic', 'content', "
        "'manual', 'local-deterministic', 64, 'cosine', 'v1', '[]'::jsonb, :hash, 'active')"
    )
    for invalid_hash in (None, "A" * 64):
        with pytest.raises(IntegrityError):
            async with clean_head.begin() as connection:
                await connection.execute(
                    statement,
                    {"id": str(uuid4()), "hash": invalid_hash},
                )


@pytest.mark.asyncio
async def test_memory_constraints_core_constraints_and_index_are_distinct(clean_head) -> None:
    async with clean_head.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT con.conrelid, rel.relname, con.conname FROM pg_constraint con "
                    "JOIN pg_class rel ON rel.oid=con.conrelid "
                    "WHERE con.conrelid IN (to_regclass('long_term_memory'), "
                    "to_regclass('agent_core_memory_block'))"
                )
            )
        ).all()
        index_flags = (
            await connection.execute(
                text(
                    "SELECT indisunique, indisvalid, indisready FROM pg_index "
                    "WHERE indexrelid=to_regclass(:index_name)"
                ),
                {"index_name": ACTIVE_EXACT_INDEX},
            )
        ).one()
    by_table = {
        table: {name for _oid, row_table, name in rows if row_table == table}
        for table in ("long_term_memory", "agent_core_memory_block")
    }
    assert M_P2_REQUIRED_CHECKS <= by_table["long_term_memory"] | by_table[
        "agent_core_memory_block"
    ]
    assert CORE_UNIQUE_CONSTRAINT in by_table["agent_core_memory_block"]
    assert index_flags == (True, True, True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_name", "drift_sql", "restore_sql"),
    [
        (
            "not-validated-check",
            (
                f"ALTER TABLE long_term_memory DROP CONSTRAINT {SCOPE_ENV_NONBLANK_CHECK}",
                f"ALTER TABLE long_term_memory ADD CONSTRAINT {SCOPE_ENV_NONBLANK_CHECK} "
                "CHECK (scope_env IS NULL OR btrim(scope_env) <> '') NOT VALID",
            ),
            (
                f"ALTER TABLE long_term_memory DROP CONSTRAINT {SCOPE_ENV_NONBLANK_CHECK}",
                f"ALTER TABLE long_term_memory ADD CONSTRAINT {SCOPE_ENV_NONBLANK_CHECK} "
                "CHECK (scope_env IS NULL OR btrim(scope_env) <> '')",
            ),
        ),
        (
            "deferrable-core-unique",
            (
                f"ALTER TABLE agent_core_memory_block DROP CONSTRAINT {CORE_UNIQUE_CONSTRAINT}",
                f"ALTER TABLE agent_core_memory_block ADD CONSTRAINT {CORE_UNIQUE_CONSTRAINT} "
                "UNIQUE (tenant_id, user_id, agent_id, block_key) DEFERRABLE INITIALLY IMMEDIATE",
            ),
            (
                f"ALTER TABLE agent_core_memory_block DROP CONSTRAINT {CORE_UNIQUE_CONSTRAINT}",
                f"ALTER TABLE agent_core_memory_block ADD CONSTRAINT {CORE_UNIQUE_CONSTRAINT} "
                "UNIQUE (tenant_id, user_id, agent_id, block_key) NOT DEFERRABLE",
            ),
        ),
        (
            "nullable-core-hash",
            (
                "ALTER TABLE agent_core_memory_block ALTER COLUMN content_hash DROP NOT NULL",
            ),
            (
                "ALTER TABLE agent_core_memory_block ALTER COLUMN content_hash SET NOT NULL",
            ),
        ),
        (
            "wrong-exact-index-cast",
            (
                f"DROP INDEX {ACTIVE_EXACT_INDEX}",
                f"CREATE UNIQUE INDEX {ACTIVE_EXACT_INDEX} ON long_term_memory "
                "(tenant_id, user_id, agent_id, type, "
                "COALESCE(scope_service::text, ''::text), "
                "COALESCE(scope_env::text, ''::text), content_hash) "
                "WHERE status = 'active'",
            ),
            (
                f"DROP INDEX {ACTIVE_EXACT_INDEX}",
                f"CREATE UNIQUE INDEX {ACTIVE_EXACT_INDEX} ON long_term_memory "
                "(tenant_id, user_id, agent_id, type, COALESCE(scope_service, ''), "
                "COALESCE(scope_env, ''), content_hash) WHERE status = 'active'",
            ),
        ),
        (
            "inverse-active-hash-check",
            (
                f"ALTER TABLE long_term_memory DROP CONSTRAINT {ACTIVE_CONTENT_HASH_CHECK}",
                f"ALTER TABLE long_term_memory ADD CONSTRAINT {ACTIVE_CONTENT_HASH_CHECK} "
                "CHECK (status <> 'active' OR (content_hash IS NOT NULL "
                "AND content_hash !~ '^[0-9a-f]{64}$'))",
            ),
            (
                f"ALTER TABLE long_term_memory DROP CONSTRAINT {ACTIVE_CONTENT_HASH_CHECK}",
                f"ALTER TABLE long_term_memory ADD CONSTRAINT {ACTIVE_CONTENT_HASH_CHECK} "
                "CHECK (status <> 'active' OR (content_hash IS NOT NULL "
                "AND content_hash ~ '^[0-9a-f]{64}$'))",
            ),
        ),
        (
            "inverse-exact-index-predicate",
            (
                f"DROP INDEX {ACTIVE_EXACT_INDEX}",
                f"CREATE UNIQUE INDEX {ACTIVE_EXACT_INDEX} ON long_term_memory "
                "(tenant_id, user_id, agent_id, type, COALESCE(scope_service, ''), "
                "COALESCE(scope_env, ''), content_hash) WHERE status <> 'active'",
            ),
            (
                f"DROP INDEX {ACTIVE_EXACT_INDEX}",
                f"CREATE UNIQUE INDEX {ACTIVE_EXACT_INDEX} ON long_term_memory "
                "(tenant_id, user_id, agent_id, type, COALESCE(scope_service, ''), "
                "COALESCE(scope_env, ''), content_hash) WHERE status = 'active'",
            ),
        ),
    ],
)
async def test_readiness_rejects_real_degraded_catalog_and_restores_it(
    clean_head,
    case_name: str,
    drift_sql: tuple[str, ...],
    restore_sql: tuple[str, ...],
) -> None:
    drift_applied = False
    try:
        async with clean_head.begin() as connection:
            for statement in drift_sql:
                await connection.execute(text(statement))
        drift_applied = True

        with pytest.raises(MemoryStoreContractError):
            await _repository(clean_head).ensure_ready()
    finally:
        if drift_applied:
            async with clean_head.begin() as connection:
                for statement in restore_sql:
                    await connection.execute(text(statement))

    await _repository(clean_head).ensure_ready()


@pytest.mark.asyncio
async def test_repository_forces_read_committed_over_higher_engine_default(clean_head) -> None:
    observed: list[str] = []
    engine = create_async_engine(
        _database_url(),
        pool_pre_ping=True,
        connect_args={
            "server_settings": {"default_transaction_isolation": "serializable"}
        },
    )

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def capture_statement(_conn, _cursor, statement, _params, _context, _many):
        if statement.startswith("SET TRANSACTION ISOLATION LEVEL"):
            observed.append(statement)

    repository = _repository(engine)
    await repository.ensure_default_core_blocks("isolation", "user", "agent")
    await repository.cas_replace_core_content(
        "isolation", "user", "agent", "user_rules", "content", expected_version=1
    )
    await repository.write_archival_exact(
        _archival("isolation", "user", "agent", content="archival")
    )
    assert observed == ["SET TRANSACTION ISOLATION LEVEL READ COMMITTED"] * 3
    await engine.dispose()


def _dirty_memory_insert(*, row_id: str, content: str, tags_json: str) -> tuple[Any, dict]:
    return (
        text(
            "INSERT INTO long_term_memory "
            "(id, tenant_id, user_id, agent_id, type, topic, content, source, "
            "embedding_model, embedding_dimension, embedding_metric, embedding_version, "
            "tags, content_hash, status) VALUES "
            "(:id, 'dirty', 'user', 'agent', 'experience', 'topic', :content, 'manual', "
            "'local-deterministic', 64, 'cosine', 'v1', CAST(:tags AS jsonb), NULL, 'active')"
        ),
        {"id": row_id, "content": content, "tags": tags_json},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_name", "expected_code"),
    [
        ("canonical-empty", "memory_migration_canonical_empty"),
        ("tags-scalar", "memory_migration_invalid_tags"),
        ("tags-non-string", "memory_migration_invalid_tags"),
        ("active-duplicates", "memory_migration_active_exact_duplicates"),
        ("core-version", "memory_migration_invalid_core_version"),
        ("core-max-tokens", "memory_migration_invalid_core_max_tokens"),
        ("core-hash", "memory_migration_invalid_core_hash"),
        ("core-blank-id", "memory_migration_invalid_core_id"),
        ("blank-id", "memory_migration_invalid_memory_id"),
    ],
)
async def test_dirty_migrations_fail_closed_without_marking_revision(
    clean_pre_mp2,
    case_name: str,
    expected_code: str,
    caplog,
    capsys,
) -> None:
    engine = clean_pre_mp2
    dirty_content = "PRIVATE-DIRTY-CONTENT"
    protected_contents: list[str] = []
    async with engine.begin() as connection:
        if case_name.startswith("core-"):
            core_content = f"PRIVATE-CORE-CONTENT-{case_name}"
            protected_contents.append(core_content)
            version = 0 if case_name == "core-version" else 1
            max_tokens = 0 if case_name == "core-max-tokens" else 100
            core_hash = (
                "invalid"
                if case_name == "core-hash"
                else hashlib.sha256(core_content.encode("utf-8")).hexdigest()
            )
            await connection.execute(
                text(
                    "INSERT INTO agent_core_memory_block "
                    "(id, tenant_id, user_id, agent_id, block_key, description, content, "
                    "max_tokens, version, read_only, source, content_hash, status) VALUES "
                    "(:id, 'dirty', 'user', 'agent', 'user_rules', 'rules', :content, "
                    ":max_tokens, :version, false, 'memory_service', :hash, 'active')"
                ),
                {
                    "id": "" if case_name == "core-blank-id" else str(uuid4()),
                    "content": core_content,
                    "max_tokens": max_tokens,
                    "version": version,
                    "hash": core_hash,
                },
            )
        else:
            contents = [dirty_content]
            tags_json = "[]"
            if case_name == "canonical-empty":
                contents = ["!!!"]
            elif case_name == "tags-scalar":
                tags_json = "{}"
            elif case_name == "tags-non-string":
                tags_json = "[1]"
            elif case_name == "active-duplicates":
                contents = [dirty_content, dirty_content]
            protected_contents.extend(contents)
            for content in contents:
                statement, params = _dirty_memory_insert(
                    row_id="" if case_name == "blank-id" else str(uuid4()),
                    content=content,
                    tags_json=tags_json,
                )
                await connection.execute(statement, params)

    memory_before = await _raw_memory_rows(engine, tenant_id="dirty")
    core_before = await _raw_core_rows(engine, tenant_id="dirty")
    caplog.clear()
    capsys.readouterr()
    unexpected_error = False
    try:
        await _upgrade("head")
    except RuntimeError as exc:
        message = str(exc)
    except Exception:
        message = ""
        unexpected_error = True
    else:
        pytest.fail("Dirty migration unexpectedly succeeded.")
    captured = capsys.readouterr()
    captured_logs = caplog.text
    caplog.clear()
    _assert_no_sensitive_or_driver_output(
        message,
        captured.out,
        captured.err,
        captured_logs,
        secrets=tuple(dict.fromkeys(protected_contents)) + _database_url_secrets(),
    )
    if unexpected_error:
        pytest.fail("Migration failure escaped its safe contract error.")
    if expected_code not in message:
        pytest.fail("Migration failed without its stable contract error code.")
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            PRE_M_P2_REVISION
        )
    memory_after = await _raw_memory_rows(engine, tenant_id="dirty")
    core_after = await _raw_core_rows(engine, tenant_id="dirty")
    if memory_after != memory_before or core_after != core_before:
        pytest.fail("Failed migration changed dirty source rows.")


@pytest.mark.asyncio
async def test_migration_normalizes_legacy_tag_values(clean_pre_mp2) -> None:
    statement, params = _dirty_memory_insert(
        row_id=str(uuid4()),
        content="valid content",
        tags_json='[" first ", "", "first", "second"]',
    )
    async with clean_pre_mp2.begin() as connection:
        await connection.execute(statement, params)
    await _upgrade("head")
    async with clean_pre_mp2.connect() as connection:
        tags = await connection.scalar(
            text("SELECT tags FROM long_term_memory WHERE tenant_id='dirty'")
        )
    assert tags == ["first", "second"]


_CHILD_SCRIPT = r"""
import asyncio
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from superbiz_agent.memory.dedup import canonical_content_hash
from superbiz_agent.memory.schemas import LongTermMemory, utc_now
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository

async def main():
    url = os.environ['M_P2_TEST_DATABASE_URL']
    mode, tag = sys.argv[1], sys.argv[2]
    barrier_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    engine = create_engine(url)
    repository = PostgresMemoryRepository(create_sessionmaker(engine))
    if mode == 'write':
        if barrier_dir is not None:
            (barrier_dir / f'ready-{tag}').touch()
            while not (barrier_dir / 'go').exists():
                await asyncio.sleep(0.01)
        content = 'subprocess persistence'
        now = utc_now()
        result = await repository.write_archival_exact(LongTermMemory(
            id=str(uuid4()), tenant_id='subprocess', user_id='user', agent_id='agent',
            session_id='session', type='experience', topic='topic', content=content,
            source='manual', embedding=[0.0] * 64, embedding_model='local-deterministic',
            embedding_dimension=64, embedding_metric='cosine', embedding_version='local-64',
            content_hash=canonical_content_hash(content), tags=[tag], created_at=now,
            updated_at=now,
        ))
        payload = {'status': result.status}
    else:
        rows = await repository.list_active_memories('subprocess', 'user', 'agent')
        payload = {'count': len(rows), 'tags': sorted(rows[0].tags) if rows else []}
    await engine.dispose()
    print(json.dumps(payload, sort_keys=True))

asyncio.run(main())
"""


async def _child(
    mode: str,
    tag: str,
    barrier_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = {
        "M_P2_TEST_DATABASE_URL": _database_url(),
        "PYTHONPATH": str(ROOT / "src"),
    }
    command_line = [sys.executable, "-c", _CHILD_SCRIPT, mode, tag]
    if barrier_dir is not None:
        command_line.append(str(barrier_dir))
    return await asyncio.to_thread(
        subprocess.run,
        command_line,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


async def _wait_for_subprocess_writers(barrier_dir: Path, tags: tuple[str, ...]) -> None:
    async def wait_until_ready() -> None:
        while not all((barrier_dir / f"ready-{tag}").exists() for tag in tags):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait_until_ready(), timeout=10)


def _assert_subprocess_succeeded(process: subprocess.CompletedProcess[str]) -> None:
    _assert_no_sensitive_or_driver_output(
        process.stdout,
        process.stderr,
        secrets=_database_url_secrets(),
    )
    if process.returncode != 0:
        pytest.fail("PostgreSQL acceptance subprocess failed without exposing its output.")


def _subprocess_json(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError:
        pytest.fail("PostgreSQL subprocess returned invalid JSON.")
    if not isinstance(payload, dict):
        pytest.fail("PostgreSQL subprocess returned a non-object JSON payload.")
    return payload


@pytest.mark.asyncio
async def test_os_subprocess_concurrency_and_cross_process_recovery(
    clean_head,
    tmp_path: Path,
) -> None:
    tags = ("process-a", "process-b")
    first_task = asyncio.create_task(_child("write", tags[0], tmp_path))
    second_task = asyncio.create_task(_child("write", tags[1], tmp_path))
    await _wait_for_subprocess_writers(tmp_path, tags)
    (tmp_path / "go").touch()
    first, second = await asyncio.gather(first_task, second_task)
    _assert_subprocess_succeeded(first)
    _assert_subprocess_succeeded(second)
    writer_payloads = [_subprocess_json(first), _subprocess_json(second)]
    if any(set(payload) != {"status"} for payload in writer_payloads):
        pytest.fail("PostgreSQL writer subprocess returned an unexpected JSON shape.")
    assert sorted(payload["status"] for payload in writer_payloads) == [
        "duplicate_skipped",
        "written",
    ]
    recovered = await _child("read", "unused")
    _assert_subprocess_succeeded(recovered)
    payload = _subprocess_json(recovered)
    assert payload == {"count": 1, "tags": ["process-a", "process-b"]}


@pytest.mark.asyncio
async def test_database_unavailable_never_falls_back_to_memory(clean_head) -> None:
    unavailable_url = str(make_url(_database_url()).set(host="127.0.0.1", port=1))
    runtime = build_memory_runtime(
        _settings(database_url=unavailable_url),
        trace_store=InMemoryRolloutEventStore(),
    )
    with pytest.raises(MemoryStoreUnavailableError):
        await runtime.ensure_ready()
    assert runtime.backend == "postgres"
    assert runtime.fixture_admin is None
    await runtime.aclose()


@pytest.mark.asyncio
async def test_real_runtime_and_harness_close_are_repeatable(clean_head) -> None:
    runtime = build_memory_runtime(
        _settings(), trace_store=InMemoryRolloutEventStore()
    )
    waiter = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0)
    waiter.cancel()
    with suppress(asyncio.CancelledError):
        await waiter
    await asyncio.gather(runtime.aclose(), runtime.aclose())
    await runtime.aclose()

    harness_runtime = build_memory_runtime(
        _settings(), trace_store=InMemoryRolloutEventStore()
    )
    service = AgentHarnessService.build_default(
        _settings(),
        trace_store=harness_runtime.trace_store,
        memory_runtime=harness_runtime,
    )
    await asyncio.gather(service.aclose(), service.aclose())
    await service.aclose()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _openai_tool_schema_hash() -> str:
    settings = Settings(
        _env_file=None,
        memory_enabled=True,
        memory_store_backend="memory",
        rollout_store_backend="memory",
        model_provider="stub",
        rag_enabled=False,
        rag_fixture_mode=True,
    )
    runtime = build_memory_runtime(settings, trace_store=InMemoryRolloutEventStore())
    definitions = build_builtin_tools(list(runtime.tools))
    payload = [
        {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.args_model.model_json_schema(by_alias=True),
            },
        }
        for definition in definitions
    ]
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_frozen_assets_and_openai_tool_schema_hashes() -> None:
    expected = {
        ROOT / "prompts" / "ops-agent-system-v3.md": (
            "84cef93089ae4932350842786ead4cf8c92df2964e05213aeae49db9fe568b49"
        ),
        ROOT / "evals" / "datasets" / "long_term_memory_v1.json": (
            "df34b4b851f89c827e2bfdf67ffcfc167a5dd3b2f349d2423b2df3926953ff0f"
        ),
        ROOT / "src" / "superbiz_agent" / "evals" / "memory_judges.py": (
            "029f18c70971146164255a93594b6d9007072a6385a36953a061b05caba302ea"
        ),
        ROOT / "src" / "superbiz_agent" / "harness" / "graph.py": (
            "4593330fbc29c9186e6192ca1f0728dfde519c83542cd65e62a67096b0124b70"
        ),
    }
    assert {path: _sha256(path) for path in expected} == expected
    assert _openai_tool_schema_hash() == (
        "eadf90360e0268fe60f0b6259cbd5285a9d00ca6900e3637b198afbc1d5ca817"
    )
