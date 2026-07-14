from __future__ import annotations

import re
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex

from superbiz_agent.memory.dedup import canonical_content_hash, canonicalize_archival_content
from superbiz_agent.memory.errors import (
    MemoryExactConflictUnresolvedError,
    MemoryStoreContractError,
)
from superbiz_agent.memory.persistence_contract import (
    ACTIVE_EXACT_INDEX,
    CORE_UNIQUE_CONSTRAINT,
    M_P2_REQUIRED_CHECKS,
)
from superbiz_agent.memory.schemas import CoreMemoryBlock, LongTermMemory, utc_now
from superbiz_agent.memory.store import content_hash
from superbiz_agent.persistence.models import AgentCoreMemoryBlock, LongTermMemory as MemoryModel
from superbiz_agent.persistence.repositories.memory import (
    PostgresMemoryRepository,
    _active_exact_conflict_target,
)


NOW = datetime(2026, 7, 14, tzinfo=timezone.utc)


class _MappingsResult:
    def __init__(self, *, one: dict[str, Any] | None = None, rows=()) -> None:
        self.one = one
        self.rows = list(rows)

    def mappings(self):
        return self

    def one_or_none(self):
        return self.one

    def all(self):
        return list(self.rows)


class _Transaction:
    def __init__(self, session: "_Session") -> None:
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, traceback):
        if exc_type is None:
            if self.session.commit_error is not None:
                raise self.session.commit_error
            self.session.commits += 1
        return False


class _Session:
    def __init__(
        self,
        *,
        results=(),
        scalars=(),
        execute_error: Exception | None = None,
        commit_error: Exception | None = None,
    ) -> None:
        self.results = list(results)
        self.scalars = list(scalars)
        self.execute_error = execute_error
        self.commit_error = commit_error
        self.statements = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def begin(self):
        return _Transaction(self)

    async def execute(self, statement, params=None):
        self.statements.append(statement)
        if self.execute_error is not None:
            raise self.execute_error
        return self.results.pop(0)

    async def scalar(self, statement, params=None):
        self.statements.append(statement)
        return self.scalars.pop(0)


class _SessionFactory:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __call__(self):
        return self.session


def _repository(session: _Session) -> PostgresMemoryRepository:
    return PostgresMemoryRepository(_SessionFactory(session))  # type: ignore[arg-type]


def _memory(**overrides: Any) -> LongTermMemory:
    content = str(overrides.pop("content", "order-service pool exhaustion resolved"))
    values = {
        "id": "memory-a",
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "agent_id": "agent-a",
        "session_id": "session-a",
        "type": "experience",
        "topic": "order-service/pool",
        "content": content,
        "embedding": [0.0] * 64,
        "embedding_model": "local-deterministic",
        "embedding_dimension": 64,
        "embedding_metric": "cosine",
        "embedding_version": "phase4-local",
        "content_hash": canonical_content_hash(content),
        "tags": ["first"],
        "scope_service": "order-service",
        "scope_env": "production",
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return LongTermMemory(**values)


def _memory_row(memory: LongTermMemory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "tenant_id": memory.tenant_id,
        "user_id": memory.user_id,
        "agent_id": memory.agent_id,
        "session_id": memory.session_id,
        "type": memory.type,
        "topic": memory.topic,
        "content": memory.content,
        "source": memory.source,
        "embedding_model": memory.embedding_model,
        "embedding_dimension": memory.embedding_dimension,
        "embedding_metric": memory.embedding_metric,
        "embedding_version": memory.embedding_version,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "usage_count": memory.usage_count,
        "last_used_at": memory.last_used_at,
        "status": memory.status,
        "archived_at": memory.archived_at,
        "archive_reason": memory.archive_reason,
        "tags": memory.tags,
        "scope_service": memory.scope_service,
        "scope_env": memory.scope_env,
        "content_hash": memory.content_hash,
    }


def _core_row(block: CoreMemoryBlock) -> dict[str, Any]:
    return {
        "id": block.id,
        "tenant_id": block.tenant_id,
        "user_id": block.user_id,
        "agent_id": block.agent_id,
        "block_key": block.block_key,
        "description": block.description,
        "content": block.content,
        "max_tokens": block.max_tokens,
        "version": block.version,
        "read_only": block.read_only,
        "source": block.source,
        "content_hash": block.content_hash,
        "created_at": block.created_at,
        "updated_at": block.updated_at,
        "status": block.status,
    }


def _sql(statement, *, literal_binds: bool = False) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": literal_binds},
        )
    )


def _migration_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "20260714_01_enforce_memory_persistence.py"
    )
    spec = spec_from_file_location("m_p2_memory_migration", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_models_freeze_exact_index_and_integrity_names() -> None:
    memory_constraints = {item.name for item in MemoryModel.__table__.constraints}
    core_constraints = {item.name for item in AgentCoreMemoryBlock.__table__.constraints}
    indexes = {item.name: item for item in MemoryModel.__table__.indexes}

    assert M_P2_REQUIRED_CHECKS.issubset(memory_constraints | core_constraints)
    assert CORE_UNIQUE_CONSTRAINT in core_constraints
    exact_index = indexes[ACTIVE_EXACT_INDEX]
    assert exact_index.unique is True
    ddl = str(CreateIndex(exact_index).compile(dialect=postgresql.dialect()))
    assert "COALESCE(scope_service, '')" in ddl
    assert "COALESCE(scope_env, '')" in ddl
    assert "WHERE status = 'active'" in ddl


@pytest.mark.asyncio
async def test_archival_insert_uses_explicit_partial_target_and_null_local_embedding() -> None:
    memory = _memory()
    session = _Session(results=[_MappingsResult(one=_memory_row(memory))])

    result = await _repository(session).write_archival_exact(memory)

    assert result.status == "written"
    assert session.commits == 1
    statement = session.statements[0]
    sql = _sql(statement)
    assert "ON CONFLICT (tenant_id, user_id, agent_id, type" in sql
    assert "coalesce(scope_service, '')" in sql
    assert "coalesce(scope_env, '')" in sql
    assert "WHERE status = 'active' DO NOTHING" in sql
    assert "ON CONFLICT DO NOTHING" not in sql
    assert statement.compile(dialect=postgresql.dialect()).params["embedding"] is None


@pytest.mark.asyncio
async def test_archival_duplicate_merges_tags_without_loading_or_updating_vector() -> None:
    existing = _memory(tags=["first", "shared"])
    merged = _memory(tags=["first", "shared", "second"])
    incoming = _memory(id="memory-new", topic="replacement", tags=["shared", "second"])
    session = _Session(
        results=[
            _MappingsResult(one=None),
            _MappingsResult(one=_memory_row(existing)),
            _MappingsResult(one=_memory_row(merged)),
        ]
    )

    result = await _repository(session).write_archival_exact(incoming)

    assert result.status == "duplicate_skipped"
    assert result.metadata_merged is True
    assert result.memory.id == existing.id
    assert result.memory.topic == existing.topic
    assert result.memory.tags == ["first", "shared", "second"]
    selected_names = {column.key for column in session.statements[1].selected_columns}
    assert "embedding" not in selected_names
    update_sql = _sql(session.statements[2])
    assert "tags=%(tags)s::JSONB" in update_sql
    # The vector column `embedding` must never be written/returned; only the
    # embedding_* scalar metadata columns may appear (e.g. in RETURNING).
    assert re.search(r"\bembedding\b", update_sql) is None
    assert session.commits == 1


@pytest.mark.asyncio
async def test_primary_key_or_other_integrity_conflict_is_not_reported_duplicate() -> None:
    session = _Session(
        execute_error=IntegrityError("safe", {}, RuntimeError("constraint"))
    )

    with pytest.raises(MemoryExactConflictUnresolvedError):
        await _repository(session).write_archival_exact(_memory())


@pytest.mark.asyncio
async def test_core_cas_statement_fences_identity_status_read_only_and_version() -> None:
    block = CoreMemoryBlock(
        id="core-a",
        tenant_id="tenant-a",
        user_id="user-a",
        agent_id="agent-a",
        block_key="user_rules",
        description="rules",
        content="new content",
        content_hash=content_hash("new content"),
        version=4,
        created_at=NOW,
        updated_at=NOW,
    )
    session = _Session(results=[_MappingsResult(one=_core_row(block))])

    result = await _repository(session).cas_replace_core_content(
        "tenant-a",
        "user-a",
        "agent-a",
        "user_rules",
        "new content",
        expected_version=3,
    )

    assert result.status == "updated"
    assert result.block is not None and result.block.version == 4
    sql = _sql(session.statements[0], literal_binds=True)
    assert "tenant_id = 'tenant-a'" in sql
    assert "user_id = 'user-a'" in sql
    assert "agent_id = 'agent-a'" in sql
    assert "status = 'active'" in sql
    assert "read_only IS false" in sql
    assert "version = 3" in sql
    assert session.commits == 1


@pytest.mark.asyncio
async def test_scalar_reads_never_select_vector_column() -> None:
    session = _Session(results=[_MappingsResult(rows=[])])

    assert await _repository(session).list_active_memories(
        "tenant-a", "user-a", "agent-a"
    ) == []

    selected_names = {column.key for column in session.statements[0].selected_columns}
    assert "embedding" not in selected_names
    assert {"tenant_id", "user_id", "agent_id"}.issubset(selected_names)


@pytest.mark.asyncio
async def test_readiness_rejects_old_schema_and_accepts_frozen_capabilities() -> None:
    old_session = _Session(results=[_MappingsResult(rows=[])])
    with pytest.raises(MemoryStoreContractError):
        await _repository(old_session).ensure_ready()

    fragments = {
        "chk_long_term_memory_active_content_hash": "CHECK (status <> 'active' OR content_hash ~ '[0-9a-f]{64}')",
        "chk_long_term_memory_scope_service_nonblank": "CHECK (scope_service IS NULL OR btrim(scope_service) <> '')",
        "chk_long_term_memory_scope_env_nonblank": "CHECK (scope_env IS NULL OR btrim(scope_env) <> '')",
        "chk_long_term_memory_tags_array": "CHECK (jsonb_typeof(tags) = 'array')",
        "chk_core_memory_block_key": "CHECK (block_key IN ('user_rules', 'user_ops_profile', 'service_notes'))",
        "chk_core_memory_version_positive": "CHECK (version >= 1)",
        "chk_core_memory_max_tokens_positive": "CHECK (max_tokens > 0)",
        "chk_core_memory_content_hash_format": "CHECK (content_hash = '' OR length(content_hash) = 64)",
    }
    rows = [
        {"table_name": "x", "conname": name, "contype": "c", "definition": definition}
        for name, definition in fragments.items()
    ]
    rows.append(
        {
            "table_name": "agent_core_memory_block",
            "conname": CORE_UNIQUE_CONSTRAINT,
            "contype": "u",
            "definition": "UNIQUE (tenant_id, user_id, agent_id, block_key)",
        }
    )
    index = (
        "CREATE UNIQUE INDEX uq_long_term_memory_active_exact ON long_term_memory "
        "(tenant_id, user_id, agent_id, type, coalesce(scope_service, ''), "
        "coalesce(scope_env, ''), content_hash) WHERE status = 'active'"
    )
    ready_session = _Session(
        results=[_MappingsResult(rows=rows)],
        scalars=[index, "vector(1024)"],
    )

    await _repository(ready_session).ensure_ready()


def test_migration_freezes_canonicalizer_names_and_single_head() -> None:
    migration = _migration_module()

    samples = [
        " Cafe\u0301\r\nincident   resolved！！！ ",
        "internal,punctuation remains;",
        "!!!",
    ]
    assert migration.revision == "20260714_01"
    assert migration.down_revision == "20260712_01"
    assert migration.ACTIVE_EXACT_INDEX == ACTIVE_EXACT_INDEX
    for sample in samples:
        assert migration._canonicalize_archival_content(sample) == (
            canonicalize_archival_content(sample)
        )
        assert migration._canonical_content_hash(sample) == canonical_content_hash(sample)


def test_local_deterministic_candidate_timestamps_are_valid() -> None:
    memory = _memory(created_at=utc_now(), updated_at=utc_now())
    assert memory.embedding_dimension == 64
    assert memory.embedding_model == "local-deterministic"


def test_conflict_target_compiles_as_frozen_expression() -> None:
    target_sql = ", ".join(_sql(expression) for expression in _active_exact_conflict_target())
    assert "coalesce(long_term_memory.scope_service, '')" in target_sql
    assert "coalesce(long_term_memory.scope_env, '')" in target_sql
