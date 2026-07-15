from __future__ import annotations

import re
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
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
    ACTIVE_CONTENT_HASH_CHECK,
    ACTIVE_EXACT_INDEX,
    CORE_UNIQUE_CONSTRAINT,
    M_P2_REQUIRED_CHECKS,
)
from superbiz_agent.memory.schemas import CoreMemoryBlock, LongTermMemory, utc_now
from superbiz_agent.memory.store import content_hash
from superbiz_agent.persistence.models import AgentCoreMemoryBlock, LongTermMemory as MemoryModel
from superbiz_agent.persistence.alembic_url import (
    EXPLICIT_DATABASE_URL_ATTRIBUTE,
    resolve_alembic_database_url,
)
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
        if str(statement).strip().upper() == "SET TRANSACTION ISOLATION LEVEL READ COMMITTED":
            return _MappingsResult()
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


def _ready_catalog_results(
    *,
    constraint_overrides: dict[str, Any] | None = None,
    index_overrides: dict[str, Any] | None = None,
    vector_overrides: dict[str, Any] | None = None,
) -> list[_MappingsResult]:
    memory_oid = 101
    core_oid = 202
    expressions = {
        "chk_long_term_memory_active_content_hash": (
            "status <> 'active'::text OR content_hash IS NOT NULL "
            "AND content_hash::text ~ '^[0-9a-f]{64}$'::text"
        ),
        "chk_long_term_memory_scope_service_nonblank": (
            "scope_service IS NULL OR btrim(scope_service::text) <> ''::text"
        ),
        "chk_long_term_memory_scope_env_nonblank": (
            "scope_env IS NULL OR btrim(scope_env::text) <> ''::text"
        ),
        "chk_long_term_memory_tags_array": "jsonb_typeof(tags) = 'array'::text",
        "chk_core_memory_block_key": (
            "block_key::text = ANY (ARRAY['user_rules'::character varying, "
            "'user_ops_profile'::character varying, "
            "'service_notes'::character varying]::text[])"
        ),
        "chk_core_memory_version_positive": "version >= 1",
        "chk_core_memory_max_tokens_positive": "max_tokens > 0",
        "chk_core_memory_content_hash_format": (
            "content_hash::text = ''::text OR "
            "content_hash::text ~ '^[0-9a-f]{64}$'::text"
        ),
    }
    memory_names = {
        "chk_long_term_memory_active_content_hash",
        "chk_long_term_memory_scope_service_nonblank",
        "chk_long_term_memory_scope_env_nonblank",
        "chk_long_term_memory_tags_array",
    }
    rows = [
        {
            "relation_oid": memory_oid if name in memory_names else core_oid,
            "conname": name,
            "contype": "c",
            "expression": expression,
            "key_columns": [],
        }
        for name, expression in expressions.items()
    ]
    rows.append(
        {
            "relation_oid": core_oid,
            "conname": CORE_UNIQUE_CONSTRAINT,
            "contype": "u",
            "expression": None,
            "key_columns": ["tenant_id", "user_id", "agent_id", "block_key"],
        }
    )
    if constraint_overrides:
        target = next(
            row for row in rows if row["conname"] == constraint_overrides["conname"]
        )
        target.update({key: value for key, value in constraint_overrides.items() if key != "conname"})
    index = {
        "relation_oid": memory_oid,
        "schema_name": "public",
        "indisunique": True,
        "indisvalid": True,
        "indisready": True,
        "indnkeyatts": 7,
        "indnatts": 7,
        "predicate": "status::text = 'active'::text",
        "key_expressions": [
            "tenant_id",
            "user_id",
            "agent_id",
            "type",
            "COALESCE(scope_service, ''::character varying)",
            "COALESCE(scope_env, ''::character varying)",
            "content_hash",
        ],
    }
    if index_overrides:
        index.update(index_overrides)
    vector = {
        "relation_oid": memory_oid,
        "attname": "embedding",
        "formatted_type": "vector(1024)",
    }
    if vector_overrides:
        vector.update(vector_overrides)
    return [
        _MappingsResult(
            one={
                "memory_oid": memory_oid,
                "memory_schema": "public",
                "core_oid": core_oid,
                "core_schema": "public",
            }
        ),
        _MappingsResult(rows=rows),
        _MappingsResult(one=index),
        _MappingsResult(one=vector),
    ]


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
    active_hash = next(
        item for item in MemoryModel.__table__.constraints if item.name == ACTIVE_CONTENT_HASH_CHECK
    )
    assert "content_hash IS NOT NULL" in str(active_hash.sqltext)
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
    assert str(session.statements[0]) == "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
    statement = session.statements[1]
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
        ],
        scalars=[False],
    )

    result = await _repository(session).write_archival_exact(incoming)

    assert result.status == "duplicate_skipped"
    assert result.metadata_merged is True
    assert result.memory.id == existing.id
    assert result.memory.topic == existing.topic
    assert result.memory.tags == ["first", "shared", "second"]
    selected_names = {column.key for column in session.statements[2].selected_columns}
    assert "embedding" not in selected_names
    update_sql = _sql(session.statements[4])
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
async def test_exact_and_unrelated_candidate_id_conflict_is_not_reported_duplicate() -> None:
    existing = _memory(id="exact-row")
    incoming = _memory(id="unrelated-row")
    session = _Session(
        results=[
            _MappingsResult(one=None),
            _MappingsResult(one=_memory_row(existing)),
        ],
        scalars=[True],
    )

    with pytest.raises(MemoryExactConflictUnresolvedError):
        await _repository(session).write_archival_exact(incoming)

    assert session.commits == 0
    id_probe_sql = _sql(session.statements[3], literal_binds=True)
    assert "long_term_memory.id = 'unrelated-row'" in id_probe_sql
    assert "content" not in {column.key for column in session.statements[3].selected_columns}


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
    assert str(session.statements[0]) == "SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
    sql = _sql(session.statements[1], literal_binds=True)
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
    old_session = _Session(results=[_MappingsResult(one=None)])
    with pytest.raises(MemoryStoreContractError):
        await _repository(old_session).ensure_ready()

    ready_session = _Session(results=_ready_catalog_results())

    await _repository(ready_session).ensure_ready()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("catalog_overrides", "kind"),
    [
        (
            {
                "conname": "chk_long_term_memory_active_content_hash",
                "relation_oid": 202,
            },
            "constraint",
        ),
        (
            {
                "conname": "chk_long_term_memory_active_content_hash",
                "expression": "status <> 'active' OR content_hash ~ '^[0-9a-f]{64}$'",
            },
            "constraint",
        ),
        ({"indisunique": False}, "index"),
        ({"indisvalid": False}, "index"),
        ({"indisready": False}, "index"),
        ({"indnkeyatts": 8, "indnatts": 8}, "index"),
        ({"predicate": "status = 'active' OR status = 'archived'"}, "index"),
        (
            {
                "key_expressions": [
                    "tenant_id",
                    "user_id",
                    "agent_id",
                    "type",
                    "scope_service",
                    "COALESCE(scope_env, '')",
                    "content_hash",
                ]
            },
            "index",
        ),
        ({"relation_oid": 202}, "vector"),
        ({"formatted_type": "vector(64)"}, "vector"),
    ],
)
async def test_readiness_rejects_wrong_relation_or_near_match_capability(
    catalog_overrides: dict[str, Any],
    kind: str,
) -> None:
    kwargs = {f"{kind}_overrides": catalog_overrides}
    session = _Session(results=_ready_catalog_results(**kwargs))

    with pytest.raises(MemoryStoreContractError):
        await _repository(session).ensure_ready()


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
    assert migration._normalize_tag_values(
        [" first ", "", "first", "second", " second "]
    ) == ["first", "second"]
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


def test_explicit_alembic_url_precedes_settings_without_calling_settings() -> None:
    config = SimpleNamespace(
        attributes={EXPLICIT_DATABASE_URL_ATTRIBUTE: "postgresql+asyncpg://isolated.invalid/test"}
    )

    def forbidden_settings():
        raise AssertionError("Settings must not be consulted for an explicit Alembic URL")

    assert resolve_alembic_database_url(
        config,
        settings_factory=forbidden_settings,
    ).endswith("/test")


def test_alembic_url_uses_settings_only_without_explicit_attribute() -> None:
    config = SimpleNamespace(attributes={})
    settings = SimpleNamespace(database_url="postgresql+asyncpg://settings.invalid/db")

    assert resolve_alembic_database_url(
        config,
        settings_factory=lambda: settings,
    ).endswith("/db")


@pytest.mark.parametrize(
    "overrides",
    [
        {"content": " !!! "},
        {"embedding": [float("nan")] * 64},
        {"embedding": [float("inf")] * 64},
        {"embedding_dimension": 0},
        {"embedding_model": ""},
        {"embedding_version": ""},
        {"source": "memory_service"},
        {"status": "archived"},
        {"type": "rule"},
        {"tags": ["valid", 1]},
        {"scope_service": 1},
    ],
)
@pytest.mark.asyncio
async def test_repository_rejects_malformed_direct_archival_inputs(
    overrides: dict[str, Any],
) -> None:
    session = _Session()

    with pytest.raises(MemoryStoreContractError):
        await _repository(session).write_archival_exact(_memory(**overrides))

    assert session.statements == []


@pytest.mark.asyncio
async def test_repository_normalizes_duplicate_blank_and_whitespace_tags() -> None:
    normalized = _memory(tags=[" first ", "", "first", "second", " second "])
    normalized.tags = ["first", "second"]
    session = _Session(results=[_MappingsResult(one=_memory_row(normalized))])

    result = await _repository(session).write_archival_exact(
        _memory(tags=[" first ", "", "first", "second", " second "])
    )

    assert result.memory.tags == ["first", "second"]
    statement = session.statements[1]
    assert statement.compile(dialect=postgresql.dialect()).params["tags"] == [
        "first",
        "second",
    ]


def test_migration_tags_preflight_uses_guarded_json_array_expansion() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "20260714_01_enforce_memory_persistence.py"
    ).read_text(encoding="utf-8")

    assert "CASE WHEN jsonb_typeof(tags) = 'array' THEN EXISTS (" in source
    assert "jsonb_array_elements(tags)" in source
    assert "ELSE TRUE END" in source
