from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.events import RolloutEvent, RolloutEventType
from superbiz_agent.harness.history import recover_active_history
from superbiz_agent.harness.locks import ConversationLockManager
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore


def _event(
    event_type: RolloutEventType,
    payload: dict,
    *,
    sequence: int,
    run_id: str = "run-history",
    tool_call_id: str | None = None,
) -> RolloutEvent:
    return RolloutEvent(
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        session_id="session",
        run_id=run_id,
        event_id=str(uuid4()),
        event_type=event_type,
        sequence=sequence,
        occurred_at=datetime.now(timezone.utc).isoformat(),
        tool_call_id=tool_call_id,
        payload=payload,
    )


@pytest.mark.asyncio
async def test_memory_rollout_event_store_filters_by_session_and_run() -> None:
    store = InMemoryRolloutEventStore()
    context = AgentRequestContext("tenant", "user", "agent", "session")
    other_context = AgentRequestContext("tenant", "user", "agent", "other-session")
    run_context = RunContext(
        request_context=context,
        run_id="run-1",
        prompt_version="prompt",
        tool_schema_version="tools",
        model_provider="stub",
    )
    other_run_context = RunContext(
        request_context=other_context,
        run_id="run-2",
        prompt_version="prompt",
        tool_schema_version="tools",
        model_provider="stub",
    )

    await store.append_event(run_context, RolloutEventType.RUN_STARTED, {"status": "started"})
    await store.append_event(other_run_context, RolloutEventType.RUN_STARTED, {"status": "started"})
    await store.append_event(run_context, RolloutEventType.RUN_COMPLETED, {"status": "success"})

    session_events = await store.list_by_session(context)
    run_events = await store.list_by_run("tenant", "run-1")

    assert [event.event_type for event in session_events] == [
        RolloutEventType.RUN_STARTED,
        RolloutEventType.RUN_COMPLETED,
    ]
    assert [event.sequence for event in session_events] == [1, 3]
    assert [event.event_type for event in run_events] == [
        RolloutEventType.RUN_STARTED,
        RolloutEventType.RUN_COMPLETED,
    ]

    await store.clear_session(context)
    assert await store.list_by_session(context) == []
    assert len(await store.list_by_session(other_context)) == 1


@pytest.mark.asyncio
async def test_memory_rollout_event_store_requires_tenant_for_run_reads() -> None:
    store = InMemoryRolloutEventStore()

    with pytest.raises(ValueError):
        await store.list_by_run("", "run-1")


def test_recover_active_history_applies_only_replayable_events() -> None:
    events = [
        _event(RolloutEventType.RUN_STARTED, {"status": "started"}, sequence=1),
        _event(RolloutEventType.USER_MESSAGE_APPENDED, {"content": "first"}, sequence=2),
        _event(
            RolloutEventType.TOOL_CALL_STARTED,
            {
                "toolName": "getCurrentDateTime",
                "arguments": {"timezone": "Asia/Shanghai"},
            },
            sequence=3,
            run_id="run-history",
            tool_call_id="toolcall-history-1",
        ),
        _event(
            RolloutEventType.TOOL_CALL_COMPLETED,
            {"toolName": "getCurrentDateTime", "result": {"success": True}},
            sequence=4,
            run_id="run-history",
            tool_call_id="toolcall-history-1",
        ),
        _event(RolloutEventType.ASSISTANT_MESSAGE_APPENDED, {"content": "answer"}, sequence=5),
        _event(RolloutEventType.RUN_FAILED, {"errorMessage": "ignored"}, sequence=6),
    ]

    history = recover_active_history(events)

    assert [(message.role, message.name, message.content) for message in history] == [
        ("user", None, "first"),
        ("assistant", None, ""),
        ("tool", "getCurrentDateTime", '{"success": true}'),
        ("assistant", None, "answer"),
    ]
    assert history[1].tool_calls[0].name == "getCurrentDateTime"
    assert history[1].tool_calls[0].arguments == {"timezone": "Asia/Shanghai"}
    assert history[2].tool_call_id == history[1].tool_calls[0].id


@pytest.mark.asyncio
async def test_runtime_recovers_previous_turn_without_repeating_current_user_message() -> None:
    service = AgentHarnessService.placeholder()
    context = AgentRequestContext("tenant-r", "user-r", "agent-r", "session-r")

    first = await service.chat(context, "hello")
    second = await service.chat(context, "second")
    events = await service.trace_store.list_by_session(context)
    second_run_events = [event for event in events if event.run_id == second.run_id]
    context_event = next(
        event for event in second_run_events if event.event_type == RolloutEventType.CONTEXT_ASSEMBLED
    )
    user_events_in_second = [
        event
        for event in second_run_events
        if event.event_type == RolloutEventType.USER_MESSAGE_APPENDED
    ]

    assert first.success is True
    assert second.success is True
    assert RolloutEventType.THREAD_RECOVERED in [event.event_type for event in second_run_events]
    assert context_event.payload["historyItemCount"] == 2
    assert context_event.payload["messageCount"] == 6
    assert context_event.payload["hasCoreMemory"] is True
    assert context_event.payload["hasMemoryIndex"] is True
    assert context_event.payload["hasMemoryMetadata"] is True
    assert user_events_in_second[0].payload["content"] == "second"
    assert second.answer == "[stub] second"


@pytest.mark.asyncio
async def test_runtime_clear_removes_replay_history() -> None:
    service = AgentHarnessService.placeholder()
    context = AgentRequestContext("tenant-clear", "user-clear", "agent-clear", "session-clear")

    await service.chat(context, "hello")
    await service.clear(context)
    result = await service.chat(context, "after clear")
    events = await service.trace_store.list_by_session(context)
    current_run_events = [event for event in events if event.run_id == result.run_id]
    context_event = next(
        event for event in current_run_events if event.event_type == RolloutEventType.CONTEXT_ASSEMBLED
    )

    assert RolloutEventType.THREAD_RECOVERED not in [
        event.event_type for event in current_run_events
    ]
    assert context_event.payload["historyItemCount"] == 0


@pytest.mark.asyncio
async def test_conversation_lock_serializes_same_key_without_blocking_other_keys() -> None:
    manager = ConversationLockManager()
    context = AgentRequestContext("tenant", "user", "agent", "same")
    other_context = AgentRequestContext("tenant", "user", "agent", "other")
    same_key_order: list[str] = []
    other_completed = asyncio.Event()

    async def same_key_task(name: str, delay: float) -> None:
        async with manager.acquire(context):
            same_key_order.append(f"{name}:start")
            await asyncio.sleep(delay)
            same_key_order.append(f"{name}:end")

    async def other_key_task() -> None:
        async with manager.acquire(other_context):
            other_completed.set()

    task_one = asyncio.create_task(same_key_task("one", 0.05))
    await asyncio.sleep(0.01)
    task_two = asyncio.create_task(same_key_task("two", 0.0))
    task_other = asyncio.create_task(other_key_task())

    await asyncio.wait_for(other_completed.wait(), timeout=1)
    await asyncio.gather(task_one, task_two, task_other)

    assert same_key_order == ["one:start", "one:end", "two:start", "two:end"]
    assert await manager.lock_for(context) is await manager.lock_for(context)
    assert await manager.lock_for(context) is not await manager.lock_for(other_context)


def test_chat_session_api_returns_message_pair_count() -> None:
    from fastapi.testclient import TestClient

    from superbiz_agent.api.app import create_app

    app = create_app()
    client = TestClient(app)
    headers = {
        "X-Tenant-Id": "tenant-api",
        "X-User-Id": "user-api",
        "X-Agent-Id": "agent-api",
    }

    client.post("/api/chat", json={"Id": "session-api", "Question": "hello"}, headers=headers)
    response = client.get("/api/chat/session/session-api", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"] == {
        "sessionId": "session-api",
        "messagePairCount": 1,
        "createTime": 0,
    }
