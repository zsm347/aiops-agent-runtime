from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import uuid4

from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.events import RolloutEvent, RolloutEventType


class InMemoryRolloutEventStore:
    """Process-local rollout event store for tests and local runs."""

    def __init__(self) -> None:
        self._events: list[RolloutEvent] = []
        self._next_sequence = 1
        self._lock = threading.Lock()

    async def append(self, event: RolloutEvent) -> RolloutEvent:
        with self._lock:
            persisted = event.model_copy(update={"sequence": self._next_sequence})
            self._events.append(persisted)
            self._next_sequence += 1
            return persisted

    async def append_event(
        self,
        run_context: RunContext,
        event_type: RolloutEventType,
        payload: dict | None = None,
        *,
        message_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> RolloutEvent:
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

    async def list_by_run(self, tenant_id: str, run_id: str) -> list[RolloutEvent]:
        if not tenant_id or not run_id:
            raise ValueError("tenant_id and run_id are required")
        with self._lock:
            return sorted(
                [
                    event
                    for event in self._events
                    if event.tenant_id == tenant_id and event.run_id == run_id
                ],
                key=lambda event: event.sequence,
            )

    async def list_by_session(self, context: AgentRequestContext) -> list[RolloutEvent]:
        with self._lock:
            return sorted(
                [
                    event
                    for event in self._events
                    if event.tenant_id == context.tenant_id
                    and event.user_id == context.user_id
                    and event.agent_id == context.agent_id
                    and event.session_id == context.session_id
                ],
                key=lambda event: event.sequence,
            )

    async def clear_session(self, context: AgentRequestContext) -> None:
        with self._lock:
            self._events = [
                event
                for event in self._events
                if not (
                    event.tenant_id == context.tenant_id
                    and event.user_id == context.user_id
                    and event.agent_id == context.agent_id
                    and event.session_id == context.session_id
                )
            ]
