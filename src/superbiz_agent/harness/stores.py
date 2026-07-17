from __future__ import annotations

import asyncio
from typing import Protocol

from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.events import RolloutEvent, RolloutEventType


class RolloutEventStore(Protocol):
    async def append(self, event: RolloutEvent) -> RolloutEvent:
        ...

    async def append_event(
        self,
        run_context: RunContext,
        event_type: RolloutEventType,
        payload: dict | None = None,
        *,
        message_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> RolloutEvent:
        ...

    async def list_by_session(self, context: AgentRequestContext) -> list[RolloutEvent]:
        ...

    async def list_by_run(self, tenant_id: str, run_id: str) -> list[RolloutEvent]:
        ...

    async def clear_session(self, context: AgentRequestContext) -> None:
        ...


class PostgresRolloutEventStore:
    def __init__(self, repository, *, engine=None) -> None:
        self.repository = repository
        self.engine = engine
        self._lifecycle_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("PostgreSQL rollout event store is closing or closed.")

    async def append(self, event: RolloutEvent) -> RolloutEvent:
        self._ensure_open()
        return await self.repository.insert(event)

    async def append_event(
        self,
        run_context: RunContext,
        event_type: RolloutEventType,
        payload: dict | None = None,
        *,
        message_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> RolloutEvent:
        self._ensure_open()
        from datetime import datetime, timezone
        from uuid import uuid4

        request_context = run_context.request_context
        event = RolloutEvent(
            tenant_id=request_context.tenant_id,
            user_id=request_context.user_id,
            agent_id=request_context.agent_id,
            session_id=request_context.session_id or "",
            run_id=run_context.run_id,
            event_id=str(uuid4()),
            event_type=event_type,
            sequence=0,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            message_id=message_id,
            tool_call_id=tool_call_id,
            payload=payload or {},
        )
        return await self.append(event)

    async def list_by_session(self, context: AgentRequestContext) -> list[RolloutEvent]:
        self._ensure_open()
        return await self.repository.list_by_session(
            tenant_id=context.tenant_id or "",
            user_id=context.user_id or "",
            agent_id=context.agent_id or "",
            session_id=context.session_id or "",
        )

    async def list_by_run(self, tenant_id: str, run_id: str) -> list[RolloutEvent]:
        self._ensure_open()
        return await self.repository.list_by_run(tenant_id=tenant_id, run_id=run_id)

    async def clear_session(self, context: AgentRequestContext) -> None:
        self._ensure_open()
        await self.repository.clear_session(
            tenant_id=context.tenant_id or "",
            user_id=context.user_id or "",
            agent_id=context.agent_id or "",
            session_id=context.session_id or "",
        )

    async def aclose(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closing = True
            task = self._close_task
            if task is None or task.done():
                task = asyncio.create_task(self._dispose_engine())
                self._close_task = task
        await asyncio.shield(task)

    async def _dispose_engine(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()
        async with self._lifecycle_lock:
            self._closed = True
