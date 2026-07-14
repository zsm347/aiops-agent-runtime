from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import UserDefinedType


revision = "20260705_02"
down_revision = "20260705_01"
branch_labels = None
depends_on = None


class PgVector(UserDefinedType):
    cache_ok = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_col_spec(self, **kw) -> str:
        return f"VECTOR({self.dimension})"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "long_term_memory",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("topic", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("embedding", PgVector(1024), nullable=True),
        sa.Column(
            "embedding_model",
            sa.String(),
            server_default=sa.text("'text-embedding-v4'"),
            nullable=False,
        ),
        sa.Column(
            "embedding_dimension",
            sa.Integer(),
            server_default=sa.text("1024"),
            nullable=False,
        ),
        sa.Column(
            "embedding_metric",
            sa.String(),
            server_default=sa.text("'cosine'"),
            nullable=False,
        ),
        sa.Column(
            "embedding_version",
            sa.String(),
            server_default=sa.text("'phase4-default'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("usage_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(), server_default=sa.text("'active'"), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archive_reason", sa.String(), nullable=True),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("scope_service", sa.String(), nullable=True),
        sa.Column("scope_env", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.CheckConstraint("type IN ('rule', 'experience', 'knowledge')", name="chk_long_term_memory_type"),
        sa.CheckConstraint(
            "source IN ('realtime', 'rollout', 'admin_config', 'manual')",
            name="chk_long_term_memory_source",
        ),
        sa.CheckConstraint("embedding_metric IN ('cosine')", name="chk_long_term_memory_metric"),
        sa.CheckConstraint("usage_count >= 0", name="chk_long_term_memory_usage_count"),
        sa.CheckConstraint("status IN ('active', 'archived')", name="chk_long_term_memory_status"),
        sa.CheckConstraint(
            "embedding_dimension > 0",
            name="chk_long_term_memory_embedding_dimension",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_memory_tenant_user_status",
        "long_term_memory",
        ["tenant_id", "user_id", "agent_id", "status"],
    )
    op.create_index(
        "idx_memory_tenant_user_type_topic",
        "long_term_memory",
        ["tenant_id", "user_id", "agent_id", "type", "topic", "status"],
    )
    op.create_index(
        "idx_memory_tenant_scope",
        "long_term_memory",
        ["tenant_id", "user_id", "agent_id", "status", "scope_service", "scope_env"],
    )
    op.create_index(
        "idx_memory_tenant_content_hash",
        "long_term_memory",
        ["tenant_id", "user_id", "agent_id", "content_hash"],
    )

    op.create_table(
        "agent_core_memory_block",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("block_key", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column("max_tokens", sa.Integer(), server_default=sa.text("500"), nullable=False),
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("read_only", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column(
            "source",
            sa.String(),
            server_default=sa.text("'memory_service'"),
            nullable=False,
        ),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("status", sa.String(), server_default=sa.text("'active'"), nullable=False),
        sa.CheckConstraint("status IN ('active', 'archived')", name="chk_core_memory_status"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "user_id",
            "agent_id",
            "block_key",
            name="uq_core_memory_block",
        ),
    )
    op.create_index(
        "idx_core_memory_tenant_user_agent",
        "agent_core_memory_block",
        ["tenant_id", "user_id", "agent_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("idx_core_memory_tenant_user_agent", table_name="agent_core_memory_block")
    op.drop_table("agent_core_memory_block")
    op.drop_index("idx_memory_tenant_content_hash", table_name="long_term_memory")
    op.drop_index("idx_memory_tenant_scope", table_name="long_term_memory")
    op.drop_index("idx_memory_tenant_user_type_topic", table_name="long_term_memory")
    op.drop_index("idx_memory_tenant_user_status", table_name="long_term_memory")
    op.drop_table("long_term_memory")
