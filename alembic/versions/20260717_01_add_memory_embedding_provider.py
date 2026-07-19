from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from superbiz_agent.memory.persistence_contract import (
    EMBEDDING_PROVIDER_NONBLANK_CHECK,
    EMBEDDING_VECTOR_IDENTITY_CHECK,
)


revision = "20260717_01"
down_revision = "20260714_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "long_term_memory",
        sa.Column(
            "embedding_provider",
            sa.String(),
            nullable=False,
            server_default=sa.text("'legacy-unknown'"),
        ),
    )
    op.execute(
        "UPDATE long_term_memory SET embedding_provider = 'local-deterministic' "
        "WHERE embedding_model = 'local-deterministic'"
    )
    op.create_check_constraint(
        EMBEDDING_PROVIDER_NONBLANK_CHECK,
        "long_term_memory",
        "btrim(embedding_provider) <> ''",
    )
    op.create_check_constraint(
        EMBEDDING_VECTOR_IDENTITY_CHECK,
        "long_term_memory",
        "embedding IS NULL OR (embedding_dimension = 1024 "
        "AND embedding_metric = 'cosine' "
        "AND btrim(embedding_provider) <> '' "
        "AND btrim(embedding_model) <> '' "
        "AND btrim(embedding_version) <> '')",
    )


def downgrade() -> None:
    op.drop_constraint(
        EMBEDDING_VECTOR_IDENTITY_CHECK,
        "long_term_memory",
        type_="check",
    )
    op.drop_constraint(
        EMBEDDING_PROVIDER_NONBLANK_CHECK,
        "long_term_memory",
        type_="check",
    )
    op.drop_column("long_term_memory", "embedding_provider")
