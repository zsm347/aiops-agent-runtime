from __future__ import annotations

from datetime import datetime

from sqlalchemy import Select, delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from superbiz_agent.harness.events import RolloutEvent, RolloutEventStoreError, RolloutEventType
from superbiz_agent.persistence.models import AgentRolloutEvent


class RolloutEventRepository:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self.sessionmaker = sessionmaker

    async def insert(self, event: RolloutEvent) -> RolloutEvent:
        self._require_filters(
            tenant_id=event.tenant_id,
            user_id=event.user_id,
            agent_id=event.agent_id,
            session_id=event.session_id,
        )
        try:
            async with self.sessionmaker() as session:
                row = AgentRolloutEvent(
                    tenant_id=event.tenant_id or "",
                    event_id=event.event_id,
                    event_type=event.event_type.value,
                    session_id=event.session_id,
                    run_id=event.run_id,
                    message_id=event.message_id,
                    tool_call_id=event.tool_call_id,
                    user_id=event.user_id or "",
                    agent_id=event.agent_id or "",
                    occurred_at=_parse_datetime(event.occurred_at),
                    payload=event.payload or {},
                )
                session.add(row)
                await session.commit()
                await session.refresh(row)
                return self._to_event(row)
        except SQLAlchemyError as exc:
            raise RolloutEventStoreError("failed to insert rollout event") from exc

    async def list_by_session(
        self,
        *,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> list[RolloutEvent]:
        self._require_filters(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        try:
            async with self.sessionmaker() as session:
                result = await session.scalars(
                    self.select_by_session_statement(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id,
                    )
                )
                return [self._to_event(row) for row in result.all()]
        except SQLAlchemyError as exc:
            raise RolloutEventStoreError("failed to list rollout events by session") from exc

    async def list_by_run(self, *, tenant_id: str, run_id: str) -> list[RolloutEvent]:
        self._require_run_filters(tenant_id=tenant_id, run_id=run_id)
        try:
            async with self.sessionmaker() as session:
                result = await session.scalars(
                    self.select_by_run_statement(tenant_id=tenant_id, run_id=run_id)
                )
                return [self._to_event(row) for row in result.all()]
        except SQLAlchemyError as exc:
            raise RolloutEventStoreError("failed to list rollout events by run") from exc

    async def clear_session(
        self,
        *,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> None:
        self._require_filters(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        try:
            async with self.sessionmaker() as session:
                await session.execute(
                    delete(AgentRolloutEvent).where(
                        AgentRolloutEvent.tenant_id == tenant_id,
                        AgentRolloutEvent.user_id == user_id,
                        AgentRolloutEvent.agent_id == agent_id,
                        AgentRolloutEvent.session_id == session_id,
                    )
                )
                await session.commit()
        except SQLAlchemyError as exc:
            raise RolloutEventStoreError("failed to clear rollout events by session") from exc

    @staticmethod
    def select_by_session_statement(
        *,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> Select[tuple[AgentRolloutEvent]]:
        RolloutEventRepository._require_filters(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        return (
            select(AgentRolloutEvent)
            .where(
                AgentRolloutEvent.tenant_id == tenant_id,
                AgentRolloutEvent.user_id == user_id,
                AgentRolloutEvent.agent_id == agent_id,
                AgentRolloutEvent.session_id == session_id,
            )
            .order_by(AgentRolloutEvent.sequence.asc())
        )

    @staticmethod
    def select_by_run_statement(
        *,
        tenant_id: str,
        run_id: str,
    ) -> Select[tuple[AgentRolloutEvent]]:
        RolloutEventRepository._require_run_filters(tenant_id=tenant_id, run_id=run_id)
        return (
            select(AgentRolloutEvent)
            .where(
                AgentRolloutEvent.tenant_id == tenant_id,
                AgentRolloutEvent.run_id == run_id,
            )
            .order_by(AgentRolloutEvent.sequence.asc())
        )

    @staticmethod
    def _require_filters(
        *,
        tenant_id: str | None,
        user_id: str | None,
        agent_id: str | None,
        session_id: str | None,
    ) -> None:
        if not tenant_id or not user_id or not agent_id or not session_id:
            raise ValueError("tenant_id, user_id, agent_id, and session_id are required")

    @staticmethod
    def _require_run_filters(*, tenant_id: str | None, run_id: str | None) -> None:
        if not tenant_id or not run_id:
            raise ValueError("tenant_id and run_id are required")

    @staticmethod
    def _to_event(row: AgentRolloutEvent) -> RolloutEvent:
        return RolloutEvent(
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            agent_id=row.agent_id,
            event_id=row.event_id,
            event_type=RolloutEventType(row.event_type),
            sequence=row.sequence,
            occurred_at=row.occurred_at.isoformat(),
            session_id=row.session_id,
            run_id=row.run_id,
            message_id=row.message_id,
            tool_call_id=row.tool_call_id,
            payload=dict(row.payload or {}),
        )


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
