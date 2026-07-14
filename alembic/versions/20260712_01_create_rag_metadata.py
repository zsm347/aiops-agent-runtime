from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260712_01"
down_revision = "20260705_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rag_knowledge_base",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), server_default=sa.text("'active'"), nullable=False),
        sa.Column("is_default", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
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
        sa.CheckConstraint(
            "status IN ('active', 'archived')",
            name="chk_rag_knowledge_base_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_rag_knowledge_base_tenant_id_id",
        ),
    )
    op.create_index(
        "uq_rag_knowledge_base_active_default",
        "rag_knowledge_base",
        ["tenant_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active' AND is_default = TRUE"),
    )
    op.create_index(
        "idx_rag_knowledge_base_tenant_status",
        "rag_knowledge_base",
        ["tenant_id", "status"],
    )

    op.create_table(
        "rag_document",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("document_name", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("parser", sa.String(), nullable=False),
        sa.Column("parser_version", sa.String(), nullable=False),
        sa.Column("embedding_provider", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding_version", sa.String(), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("chunk_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("claim_token", sa.String(length=36), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_code", sa.String(length=100), nullable=True),
        sa.Column("failure_message", sa.String(length=1000), nullable=True),
        sa.Column("created_by", sa.String(), nullable=False),
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
        sa.CheckConstraint(
            "status IN ('indexing', 'active', 'failed', 'archived')",
            name="chk_rag_document_status",
        ),
        sa.CheckConstraint("chunk_count >= 0", name="chk_rag_document_chunk_count"),
        sa.CheckConstraint(
            "embedding_dimension > 0",
            name="chk_rag_document_embedding_dimension",
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="chk_rag_document_content_hash_sha256",
        ),
        sa.CheckConstraint(
            "((status = 'indexing' AND claim_token IS NOT NULL AND claimed_at IS NOT NULL) "
            "OR (status <> 'indexing' AND claim_token IS NULL AND claimed_at IS NULL))",
            name="chk_rag_document_claim_state",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "knowledge_base_id"],
            ["rag_knowledge_base.tenant_id", "rag_knowledge_base.id"],
            name="fk_rag_document_tenant_knowledge_base",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "knowledge_base_id",
            "content_hash",
            name="uq_rag_document_scope_content_hash",
        ),
    )
    op.create_index(
        "idx_rag_document_tenant_status",
        "rag_document",
        ["tenant_id", "status"],
    )
    op.create_index(
        "idx_rag_document_tenant_knowledge_base_status",
        "rag_document",
        ["tenant_id", "knowledge_base_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_rag_document_tenant_knowledge_base_status",
        table_name="rag_document",
    )
    op.drop_index("idx_rag_document_tenant_status", table_name="rag_document")
    op.drop_table("rag_document")
    op.drop_index(
        "idx_rag_knowledge_base_tenant_status",
        table_name="rag_knowledge_base",
    )
    op.drop_index(
        "uq_rag_knowledge_base_active_default",
        table_name="rag_knowledge_base",
    )
    op.drop_table("rag_knowledge_base")
