from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260705_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_rollout_event",
        sa.Column("sequence", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("message_id", sa.String(), nullable=True),
        sa.Column("tool_call_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("sequence"),
        sa.UniqueConstraint("event_id"),
    )
    op.create_index(
        "idx_rollout_tenant_session_sequence",
        "agent_rollout_event",
        ["tenant_id", "user_id", "agent_id", "session_id", "sequence"],
    )
    op.create_index(
        "idx_rollout_tenant_run",
        "agent_rollout_event",
        ["tenant_id", "run_id"],
    )
    op.create_index(
        "idx_rollout_event_type",
        "agent_rollout_event",
        ["event_type"],
    )


def downgrade() -> None:
    op.drop_index("idx_rollout_event_type", table_name="agent_rollout_event")
    op.drop_index("idx_rollout_tenant_run", table_name="agent_rollout_event")
    op.drop_index("idx_rollout_tenant_session_sequence", table_name="agent_rollout_event")
    op.drop_table("agent_rollout_event")
