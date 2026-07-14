from __future__ import annotations

import hashlib
import re
import unicodedata

from alembic import op
import sqlalchemy as sa


revision = "20260714_01"
down_revision = "20260712_01"
branch_labels = None
depends_on = None


ACTIVE_EXACT_INDEX = "uq_long_term_memory_active_exact"
ACTIVE_CONTENT_HASH_CHECK = "chk_long_term_memory_active_content_hash"
SCOPE_SERVICE_NONBLANK_CHECK = "chk_long_term_memory_scope_service_nonblank"
SCOPE_ENV_NONBLANK_CHECK = "chk_long_term_memory_scope_env_nonblank"
TAGS_ARRAY_CHECK = "chk_long_term_memory_tags_array"
CORE_BLOCK_KEY_CHECK = "chk_core_memory_block_key"
CORE_VERSION_POSITIVE_CHECK = "chk_core_memory_version_positive"
CORE_MAX_TOKENS_POSITIVE_CHECK = "chk_core_memory_max_tokens_positive"
CORE_CONTENT_HASH_FORMAT_CHECK = "chk_core_memory_content_hash_format"

_BATCH_SIZE = 500
_ERROR_ID_LIMIT = 10
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_TERMINATOR_RE = re.compile(r"[.!?。！？…]+$")


def _canonicalize_archival_content(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return _TRAILING_TERMINATOR_RE.sub("", normalized).rstrip()


def _canonical_content_hash(content: str) -> str:
    canonical = _canonicalize_archival_content(content)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _core_content_hash(content: str | None) -> str:
    if content is None or not content.strip():
        return ""
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _safe_failure(code: str, *, count: int, row_ids: list[str]) -> RuntimeError:
    ids = ",".join(row_ids[:_ERROR_ID_LIMIT]) or "none"
    return RuntimeError(f"{code}: count={count}; row_ids={ids}")


def _normalize_scopes(connection: sa.Connection) -> None:
    last_id = ""
    while True:
        rows = connection.execute(
            sa.text(
                "SELECT id, scope_service, scope_env FROM long_term_memory "
                "WHERE id > :last_id ORDER BY id LIMIT :batch_size"
            ),
            {"last_id": last_id, "batch_size": _BATCH_SIZE},
        ).mappings().all()
        if not rows:
            return
        for row in rows:
            service = row["scope_service"]
            env = row["scope_env"]
            normalized_service = service.strip() or None if service is not None else None
            normalized_env = env.strip() or None if env is not None else None
            if normalized_service != service or normalized_env != env:
                connection.execute(
                    sa.text(
                        "UPDATE long_term_memory "
                        "SET scope_service = :scope_service, scope_env = :scope_env "
                        "WHERE id = :id"
                    ),
                    {
                        "id": row["id"],
                        "scope_service": normalized_service,
                        "scope_env": normalized_env,
                    },
                )
        last_id = str(rows[-1]["id"])


def _backfill_archival_hashes(connection: sa.Connection) -> None:
    invalid_count = 0
    invalid_ids: list[str] = []
    last_id = ""
    while True:
        rows = connection.execute(
            sa.text(
                "SELECT id, content FROM long_term_memory "
                "WHERE status = 'active' AND id > :last_id ORDER BY id LIMIT :batch_size"
            ),
            {"last_id": last_id, "batch_size": _BATCH_SIZE},
        ).mappings().all()
        if not rows:
            break
        for row in rows:
            canonical = _canonicalize_archival_content(row["content"])
            if not canonical:
                invalid_count += 1
                if len(invalid_ids) < _ERROR_ID_LIMIT:
                    invalid_ids.append(str(row["id"]))
                continue
            connection.execute(
                sa.text("UPDATE long_term_memory SET content_hash = :hash WHERE id = :id"),
                {"id": row["id"], "hash": _canonical_content_hash(row["content"])},
            )
        last_id = str(rows[-1]["id"])
    if invalid_count:
        raise _safe_failure(
            "memory_migration_canonical_empty",
            count=invalid_count,
            row_ids=invalid_ids,
        )


def _backfill_core_hashes(connection: sa.Connection) -> None:
    last_id = ""
    while True:
        rows = connection.execute(
            sa.text(
                "SELECT id, content FROM agent_core_memory_block "
                "WHERE id > :last_id ORDER BY id LIMIT :batch_size"
            ),
            {"last_id": last_id, "batch_size": _BATCH_SIZE},
        ).mappings().all()
        if not rows:
            return
        for row in rows:
            connection.execute(
                sa.text(
                    "UPDATE agent_core_memory_block SET content_hash = :hash WHERE id = :id"
                ),
                {"id": row["id"], "hash": _core_content_hash(row["content"])},
            )
        last_id = str(rows[-1]["id"])


def _invalid_rows(
    connection: sa.Connection,
    *,
    table: str,
    predicate: str,
    code: str,
) -> None:
    count = int(
        connection.scalar(sa.text(f"SELECT count(*) FROM {table} WHERE {predicate}")) or 0
    )
    if not count:
        return
    rows = connection.execute(
        sa.text(
            f"SELECT id FROM {table} WHERE {predicate} ORDER BY id LIMIT {_ERROR_ID_LIMIT}"
        )
    ).scalars().all()
    raise _safe_failure(code, count=count, row_ids=[str(row_id) for row_id in rows])


def _preflight(connection: sa.Connection) -> None:
    _invalid_rows(
        connection,
        table="long_term_memory",
        predicate=(
            "jsonb_typeof(tags) <> 'array' OR EXISTS ("
            "SELECT 1 FROM jsonb_array_elements(tags) AS item "
            "WHERE jsonb_typeof(item) <> 'string')"
        ),
        code="memory_migration_invalid_tags",
    )
    duplicate_groups = int(
        connection.scalar(
            sa.text(
                "SELECT count(*) FROM ("
                "SELECT 1 FROM long_term_memory WHERE status = 'active' "
                "GROUP BY tenant_id, user_id, agent_id, type, "
                "COALESCE(scope_service, ''), COALESCE(scope_env, ''), content_hash "
                "HAVING count(*) > 1) AS duplicate_groups"
            )
        )
        or 0
    )
    if duplicate_groups:
        rows = connection.execute(
            sa.text(
                "SELECT min(id) FROM long_term_memory WHERE status = 'active' "
                "GROUP BY tenant_id, user_id, agent_id, type, "
                "COALESCE(scope_service, ''), COALESCE(scope_env, ''), content_hash "
                "HAVING count(*) > 1 ORDER BY min(id) LIMIT :limit"
            ),
            {"limit": _ERROR_ID_LIMIT},
        ).scalars().all()
        raise _safe_failure(
            "memory_migration_active_exact_duplicates",
            count=duplicate_groups,
            row_ids=[str(row_id) for row_id in rows],
        )
    _invalid_rows(
        connection,
        table="agent_core_memory_block",
        predicate="block_key NOT IN ('user_rules', 'user_ops_profile', 'service_notes')",
        code="memory_migration_invalid_core_block_key",
    )
    _invalid_rows(
        connection,
        table="agent_core_memory_block",
        predicate="version < 1",
        code="memory_migration_invalid_core_version",
    )
    _invalid_rows(
        connection,
        table="agent_core_memory_block",
        predicate="max_tokens <= 0",
        code="memory_migration_invalid_core_max_tokens",
    )
    _invalid_rows(
        connection,
        table="agent_core_memory_block",
        predicate="content_hash IS NULL OR content_hash !~ '^(|[0-9a-f]{64})$'",
        code="memory_migration_invalid_core_hash",
    )


def upgrade() -> None:
    connection = op.get_bind()
    _normalize_scopes(connection)
    _backfill_archival_hashes(connection)
    _backfill_core_hashes(connection)
    _preflight(connection)

    op.alter_column(
        "agent_core_memory_block",
        "content_hash",
        existing_type=sa.String(),
        nullable=False,
    )
    op.create_check_constraint(
        ACTIVE_CONTENT_HASH_CHECK,
        "long_term_memory",
        "status <> 'active' OR content_hash ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        SCOPE_SERVICE_NONBLANK_CHECK,
        "long_term_memory",
        "scope_service IS NULL OR btrim(scope_service) <> ''",
    )
    op.create_check_constraint(
        SCOPE_ENV_NONBLANK_CHECK,
        "long_term_memory",
        "scope_env IS NULL OR btrim(scope_env) <> ''",
    )
    op.create_check_constraint(
        TAGS_ARRAY_CHECK,
        "long_term_memory",
        "jsonb_typeof(tags) = 'array'",
    )
    op.create_check_constraint(
        CORE_BLOCK_KEY_CHECK,
        "agent_core_memory_block",
        "block_key IN ('user_rules', 'user_ops_profile', 'service_notes')",
    )
    op.create_check_constraint(
        CORE_VERSION_POSITIVE_CHECK,
        "agent_core_memory_block",
        "version >= 1",
    )
    op.create_check_constraint(
        CORE_MAX_TOKENS_POSITIVE_CHECK,
        "agent_core_memory_block",
        "max_tokens > 0",
    )
    op.create_check_constraint(
        CORE_CONTENT_HASH_FORMAT_CHECK,
        "agent_core_memory_block",
        "content_hash = '' OR content_hash ~ '^[0-9a-f]{64}$'",
    )
    op.create_index(
        ACTIVE_EXACT_INDEX,
        "long_term_memory",
        [
            "tenant_id",
            "user_id",
            "agent_id",
            "type",
            sa.text("COALESCE(scope_service, '')"),
            sa.text("COALESCE(scope_env, '')"),
            "content_hash",
        ],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index(ACTIVE_EXACT_INDEX, table_name="long_term_memory")
    op.drop_constraint(
        CORE_CONTENT_HASH_FORMAT_CHECK,
        "agent_core_memory_block",
        type_="check",
    )
    op.drop_constraint(
        CORE_MAX_TOKENS_POSITIVE_CHECK,
        "agent_core_memory_block",
        type_="check",
    )
    op.drop_constraint(
        CORE_VERSION_POSITIVE_CHECK,
        "agent_core_memory_block",
        type_="check",
    )
    op.drop_constraint(CORE_BLOCK_KEY_CHECK, "agent_core_memory_block", type_="check")
    op.drop_constraint(TAGS_ARRAY_CHECK, "long_term_memory", type_="check")
    op.drop_constraint(SCOPE_ENV_NONBLANK_CHECK, "long_term_memory", type_="check")
    op.drop_constraint(SCOPE_SERVICE_NONBLANK_CHECK, "long_term_memory", type_="check")
    op.drop_constraint(ACTIVE_CONTENT_HASH_CHECK, "long_term_memory", type_="check")
    op.alter_column(
        "agent_core_memory_block",
        "content_hash",
        existing_type=sa.String(),
        nullable=True,
    )
    # Scope/hash normalization is intentionally not reversed.
