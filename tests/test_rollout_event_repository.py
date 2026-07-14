from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
postgresql = pytest.importorskip("sqlalchemy.dialects.postgresql")
delete = sqlalchemy.delete

from superbiz_agent.persistence.models import AgentRolloutEvent
from superbiz_agent.persistence.repositories.rollout_events import RolloutEventRepository


def _compiled(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def test_session_statement_requires_and_applies_full_conversation_filters() -> None:
    statement = RolloutEventRepository.select_by_session_statement(
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        session_id="session",
    )
    sql = _compiled(statement)

    assert "agent_rollout_event.tenant_id = 'tenant'" in sql
    assert "agent_rollout_event.user_id = 'user'" in sql
    assert "agent_rollout_event.agent_id = 'agent'" in sql
    assert "agent_rollout_event.session_id = 'session'" in sql
    assert "ORDER BY agent_rollout_event.sequence ASC" in sql

    with pytest.raises(ValueError):
        RolloutEventRepository.select_by_session_statement(
            tenant_id="tenant",
            user_id="",
            agent_id="agent",
            session_id="session",
        )


def test_run_statement_requires_and_applies_tenant_filter() -> None:
    statement = RolloutEventRepository.select_by_run_statement(
        tenant_id="tenant",
        run_id="run",
    )
    sql = _compiled(statement)

    assert "agent_rollout_event.tenant_id = 'tenant'" in sql
    assert "agent_rollout_event.run_id = 'run'" in sql
    assert "ORDER BY agent_rollout_event.sequence ASC" in sql

    with pytest.raises(ValueError):
        RolloutEventRepository.select_by_run_statement(tenant_id="", run_id="run")


def test_clear_statement_shape_uses_full_conversation_filters() -> None:
    statement = delete(AgentRolloutEvent).where(
        AgentRolloutEvent.tenant_id == "tenant",
        AgentRolloutEvent.user_id == "user",
        AgentRolloutEvent.agent_id == "agent",
        AgentRolloutEvent.session_id == "session",
    )
    sql = _compiled(statement)

    assert "DELETE FROM agent_rollout_event" in sql
    assert "agent_rollout_event.tenant_id = 'tenant'" in sql
    assert "agent_rollout_event.user_id = 'user'" in sql
    assert "agent_rollout_event.agent_id = 'agent'" in sql
    assert "agent_rollout_event.session_id = 'session'" in sql
