from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import VECTOR

from superbiz_agent.memory.persistence_contract import (
    ACTIVE_CONTENT_HASH_CHECK,
    ACTIVE_EXACT_INDEX,
    ACTIVE_EXACT_PREDICATE_SQL,
    CORE_BLOCK_KEY_CHECK,
    CORE_CONTENT_HASH_FORMAT_CHECK,
    CORE_MAX_TOKENS_POSITIVE_CHECK,
    CORE_VERSION_POSITIVE_CHECK,
    EMBEDDING_PROVIDER_NONBLANK_CHECK,
    EMBEDDING_VECTOR_IDENTITY_CHECK,
    SCOPE_ENV_NONBLANK_CHECK,
    SCOPE_SERVICE_NONBLANK_CHECK,
    TAGS_ARRAY_CHECK,
)


class Base(DeclarativeBase):
    pass


class AgentRolloutEvent(Base):
    __tablename__ = "agent_rollout_event"
    __table_args__ = (
        Index(
            "idx_rollout_tenant_session_sequence",
            "tenant_id",
            "user_id",
            "agent_id",
            "session_id",
            "sequence",
        ),
        Index("idx_rollout_tenant_run", "tenant_id", "run_id"),
        Index("idx_rollout_event_type", "event_type"),
    )

    sequence: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    event_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String, nullable=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"
    __table_args__ = (
        CheckConstraint("type IN ('rule', 'experience', 'knowledge')", name="chk_long_term_memory_type"),
        CheckConstraint(
            "source IN ('realtime', 'rollout', 'admin_config', 'manual')",
            name="chk_long_term_memory_source",
        ),
        CheckConstraint("embedding_metric IN ('cosine')", name="chk_long_term_memory_metric"),
        CheckConstraint("usage_count >= 0", name="chk_long_term_memory_usage_count"),
        CheckConstraint("status IN ('active', 'archived')", name="chk_long_term_memory_status"),
        CheckConstraint("embedding_dimension > 0", name="chk_long_term_memory_embedding_dimension"),
        CheckConstraint(
            "status <> 'active' OR "
            "(content_hash IS NOT NULL AND content_hash ~ '^[0-9a-f]{64}$')",
            name=ACTIVE_CONTENT_HASH_CHECK,
        ),
        CheckConstraint(
            "scope_service IS NULL OR btrim(scope_service) <> ''",
            name=SCOPE_SERVICE_NONBLANK_CHECK,
        ),
        CheckConstraint(
            "scope_env IS NULL OR btrim(scope_env) <> ''",
            name=SCOPE_ENV_NONBLANK_CHECK,
        ),
        CheckConstraint("jsonb_typeof(tags) = 'array'", name=TAGS_ARRAY_CHECK),
        CheckConstraint(
            "btrim(embedding_provider) <> ''",
            name=EMBEDDING_PROVIDER_NONBLANK_CHECK,
        ),
        CheckConstraint(
            "embedding IS NULL OR (embedding_dimension = 1024 "
            "AND embedding_metric = 'cosine' "
            "AND btrim(embedding_provider) <> '' "
            "AND btrim(embedding_model) <> '' "
            "AND btrim(embedding_version) <> '')",
            name=EMBEDDING_VECTOR_IDENTITY_CHECK,
        ),
        Index(
            ACTIVE_EXACT_INDEX,
            "tenant_id",
            "user_id",
            "agent_id",
            "type",
            text("COALESCE(scope_service, '')"),
            text("COALESCE(scope_env, '')"),
            "content_hash",
            unique=True,
            postgresql_where=text(ACTIVE_EXACT_PREDICATE_SQL),
        ),
        Index("idx_memory_tenant_user_status", "tenant_id", "user_id", "agent_id", "status"),
        Index("idx_memory_tenant_user_type_topic", "tenant_id", "user_id", "agent_id", "type", "topic", "status"),
        Index("idx_memory_tenant_scope", "tenant_id", "user_id", "agent_id", "status", "scope_service", "scope_env"),
        Index("idx_memory_tenant_content_hash", "tenant_id", "user_id", "agent_id", "content_hash"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    topic: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(VECTOR(1024), nullable=True)
    embedding_provider: Mapped[str] = mapped_column(
        String,
        nullable=False,
        server_default=text("'legacy-unknown'"),
    )
    embedding_model: Mapped[str] = mapped_column(
        String,
        nullable=False,
        server_default=text("'text-embedding-v4'"),
    )
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1024"))
    embedding_metric: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'cosine'"))
    embedding_version: Mapped[str] = mapped_column(
        String,
        nullable=False,
        server_default=text("'phase4-default'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'active'"))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    archive_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    scope_service: Mapped[str | None] = mapped_column(String, nullable=True)
    scope_env: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)


class AgentCoreMemoryBlock(Base):
    __tablename__ = "agent_core_memory_block"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "user_id",
            "agent_id",
            "block_key",
            name="uq_core_memory_block",
        ),
        CheckConstraint("status IN ('active', 'archived')", name="chk_core_memory_status"),
        CheckConstraint(
            "block_key IN ('user_rules', 'user_ops_profile', 'service_notes')",
            name=CORE_BLOCK_KEY_CHECK,
        ),
        CheckConstraint("version >= 1", name=CORE_VERSION_POSITIVE_CHECK),
        CheckConstraint("max_tokens > 0", name=CORE_MAX_TOKENS_POSITIVE_CHECK),
        CheckConstraint(
            "content_hash IS NOT NULL AND "
            "(content_hash = '' OR content_hash ~ '^[0-9a-f]{64}$')",
            name=CORE_CONTENT_HASH_FORMAT_CHECK,
        ),
        Index(
            "idx_core_memory_tenant_user_agent",
            "tenant_id",
            "user_id",
            "agent_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    block_key: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("500"))
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    read_only: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("FALSE"))
    source: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'memory_service'"))
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    status: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'active'"))


class RagKnowledgeBase(Base):
    __tablename__ = "rag_knowledge_base"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id", name="uq_rag_knowledge_base_tenant_id_id"),
        CheckConstraint(
            "status IN ('active', 'archived')",
            name="chk_rag_knowledge_base_status",
        ),
        Index(
            "uq_rag_knowledge_base_active_default",
            "tenant_id",
            unique=True,
            postgresql_where=text("status = 'active' AND is_default = TRUE"),
        ),
        Index("idx_rag_knowledge_base_tenant_status", "tenant_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default=text("'active'"))
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("FALSE"))
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )


class RagDocument(Base):
    __tablename__ = "rag_document"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "knowledge_base_id"],
            ["rag_knowledge_base.tenant_id", "rag_knowledge_base.id"],
            name="fk_rag_document_tenant_knowledge_base",
        ),
        UniqueConstraint(
            "tenant_id",
            "knowledge_base_id",
            "content_hash",
            name="uq_rag_document_scope_content_hash",
        ),
        CheckConstraint(
            "status IN ('indexing', 'active', 'failed', 'archived')",
            name="chk_rag_document_status",
        ),
        CheckConstraint("chunk_count >= 0", name="chk_rag_document_chunk_count"),
        CheckConstraint(
            "embedding_dimension > 0",
            name="chk_rag_document_embedding_dimension",
        ),
        CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="chk_rag_document_content_hash_sha256",
        ),
        CheckConstraint(
            "((status = 'indexing' AND claim_token IS NOT NULL AND claimed_at IS NOT NULL) "
            "OR (status <> 'indexing' AND claim_token IS NULL AND claimed_at IS NULL))",
            name="chk_rag_document_claim_state",
        ),
        Index("idx_rag_document_tenant_status", "tenant_id", "status"),
        Index(
            "idx_rag_document_tenant_knowledge_base_status",
            "tenant_id",
            "knowledge_base_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    document_name: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    parser: Mapped[str] = mapped_column(String, nullable=False)
    parser_version: Mapped[str] = mapped_column(String, nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    embedding_version: Mapped[str] = mapped_column(String, nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    claim_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
