from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from superbiz_agent.config import Settings
from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.context_budget import ContextBudget
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.graph import AgentStreamEvent, SkeletonAgentGraph
from superbiz_agent.harness.history import recover_active_history
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.harness.token_estimator import ApproxTokenEstimator
from superbiz_agent.harness.tool_result_reducer import ToolResultReducer
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore
from superbiz_agent.harness.content_compression import DeterministicContentCompressionBackend
from superbiz_agent.model_gateway.base import ModelMessage, ModelResponse, ModelToolCall
from superbiz_agent.security.permissions import LOCAL_DEFAULT_PERMISSIONS
from superbiz_agent.tools.gateway import ToolGateway
from superbiz_agent.tools.policies import ToolPolicy
from superbiz_agent.tools.registry import ToolDefinition, ToolRegistry


class EmptyArgs(BaseModel):
    pass


class ScriptedModelGateway:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[ModelMessage], list[ToolDefinition] | None]] = []

    async def complete(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        self.calls.append((list(messages), tools))
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


def _run_context(run_id: str = "run-react") -> RunContext:
    return RunContext(
        request_context=AgentRequestContext(
            "tenant",
            "user",
            "agent",
            "session",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        ),
        run_id=run_id,
        prompt_version="ops-agent-system-v2",
        tool_schema_version="ops-tools-v1",
        model_provider="scripted",
    )


def _tool(
    name: str,
    handler,
    *,
    idempotent: bool = False,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Run {name}.",
        args_model=EmptyArgs,
        handler=handler,
        policy=ToolPolicy(tool_name=name, idempotent=idempotent),
    )


def _graph(
    responses: list[ModelResponse],
    definitions: list[ToolDefinition],
    *,
    max_rounds: int = 4,
    max_calls: int = 8,
    reducer: ToolResultReducer | None = None,
) -> tuple[SkeletonAgentGraph, ScriptedModelGateway, InMemoryRolloutEventStore]:
    trace_store = InMemoryRolloutEventStore()
    registry = ToolRegistry()
    for definition in definitions:
        registry.register(definition)
    model = ScriptedModelGateway(responses)
    graph = SkeletonAgentGraph(
        model_gateway=model,
        tool_gateway=ToolGateway(registry, trace_store),
        trace_store=trace_store,
        tool_result_reducer=reducer,
        max_tool_rounds=max_rounds,
        max_tool_calls_per_run=max_calls,
    )
    return graph, model, trace_store


@pytest.mark.asyncio
async def test_one_response_executes_all_tools_in_order_and_pairs_messages() -> None:
    executed: list[str] = []

    def handle_a(_: EmptyArgs) -> dict[str, Any]:
        executed.append("toolA")
        return {"success": True, "value": "a"}

    def handle_b(_: EmptyArgs) -> dict[str, Any]:
        executed.append("toolB")
        return {"success": True, "value": "b"}

    graph, model, trace_store = _graph(
        [
            ModelResponse(
                content="checking",
                tool_calls=[
                    ModelToolCall(id="call-a", name="toolA"),
                    ModelToolCall(id="call-b", name="toolB"),
                ],
            ),
            ModelResponse(content="done"),
        ],
        [_tool("toolA", handle_a), _tool("toolB", handle_b)],
    )

    answer = await graph.run(_run_context(), [ModelMessage(role="user", content="go")])

    assert answer == "done"
    assert executed == ["toolA", "toolB"]
    second_messages = model.calls[1][0]
    assistant = next(message for message in second_messages if message.role == "assistant")
    tools = [message for message in second_messages if message.role == "tool"]
    assert [call.id for call in assistant.tool_calls] == ["call-a", "call-b"]
    assert [message.tool_call_id for message in tools] == ["call-a", "call-b"]

    events = await trace_store.list_by_run("tenant", "run-react")
    tool_events = [
        (event.event_type, event.tool_call_id)
        for event in events
        if event.event_type
        in {RolloutEventType.TOOL_CALL_STARTED, RolloutEventType.TOOL_CALL_COMPLETED}
    ]
    assert tool_events == [
        (RolloutEventType.TOOL_CALL_STARTED, "call-a"),
        (RolloutEventType.TOOL_CALL_COMPLETED, "call-a"),
        (RolloutEventType.TOOL_CALL_STARTED, "call-b"),
        (RolloutEventType.TOOL_CALL_COMPLETED, "call-b"),
    ]


@pytest.mark.asyncio
async def test_two_consecutive_tool_rounds_complete() -> None:
    executed: list[str] = []

    def handler(_: EmptyArgs) -> dict[str, Any]:
        executed.append("ran")
        return {"success": True}

    graph, model, _ = _graph(
        [
            ModelResponse(tool_calls=[ModelToolCall(id="round-1", name="toolA")], content=""),
            ModelResponse(tool_calls=[ModelToolCall(id="round-2", name="toolA")], content=""),
            ModelResponse(content="final"),
        ],
        [_tool("toolA", handler)],
    )

    assert await graph.run(_run_context(), [ModelMessage(role="user", content="go")]) == "final"
    assert executed == ["ran", "ran"]
    assert len(model.calls) == 3
    assert all(tools for _, tools in model.calls)


@pytest.mark.asyncio
async def test_first_tool_failure_does_not_skip_later_tool_in_batch() -> None:
    executed: list[str] = []

    def fail(_: EmptyArgs) -> dict[str, Any]:
        executed.append("failed")
        raise RuntimeError("upstream exploded")

    def succeed(_: EmptyArgs) -> dict[str, Any]:
        executed.append("succeeded")
        return {"success": True}

    graph, model, _ = _graph(
        [
            ModelResponse(
                tool_calls=[
                    ModelToolCall(id="failed", name="failTool"),
                    ModelToolCall(id="succeeded", name="okTool"),
                ],
                content="",
            ),
            ModelResponse(content="handled"),
        ],
        [_tool("failTool", fail), _tool("okTool", succeed)],
    )

    assert await graph.run(_run_context(), [ModelMessage(role="user", content="go")]) == "handled"
    assert executed == ["failed", "succeeded"]
    tool_payloads = [
        json.loads(message.content) for message in model.calls[1][0] if message.role == "tool"
    ]
    assert tool_payloads[0]["success"] is False
    assert tool_payloads[1]["success"] is True
    assert graph.tool_gateway._failure_records == {}
    assert graph.tool_gateway._validation_counts == {}


@pytest.mark.asyncio
async def test_multi_tool_trace_replay_keeps_every_call_paired() -> None:
    graph, _, trace_store = _graph(
        [
            ModelResponse(
                tool_calls=[
                    ModelToolCall(id="replay-a", name="toolA"),
                    ModelToolCall(id="replay-b", name="toolB"),
                ],
                content="",
            ),
            ModelResponse(content="done"),
        ],
        [
            _tool("toolA", lambda _: {"success": True, "value": "a"}),
            _tool("toolB", lambda _: {"success": True, "value": "b"}),
        ],
    )
    run_context = _run_context()

    await graph.run(run_context, [ModelMessage(role="user", content="go")])
    events = await trace_store.list_by_run("tenant", run_context.run_id)
    recovered = recover_active_history(events)

    assistant_calls = [
        message.tool_calls[0].id
        for message in recovered
        if message.role == "assistant" and message.tool_calls
    ]
    tool_results = [
        message.tool_call_id for message in recovered if message.role == "tool"
    ]
    assert assistant_calls == ["replay-a", "replay-b"]
    assert tool_results == assistant_calls


@pytest.mark.asyncio
async def test_call_budget_blocks_entire_batch_without_side_effects() -> None:
    executed: list[str] = []

    def handler(_: EmptyArgs) -> dict[str, Any]:
        executed.append("side-effect")
        return {"success": True}

    graph, model, trace_store = _graph(
        [
            ModelResponse(
                tool_calls=[
                    ModelToolCall(id="call-1", name="writeTool"),
                    ModelToolCall(id="call-2", name="writeTool"),
                ],
                content="",
            ),
            ModelResponse(content="budget final"),
        ],
        [_tool("writeTool", handler)],
        max_calls=1,
    )

    answer = await graph.run(_run_context(), [ModelMessage(role="user", content="go")])

    assert answer == "budget final"
    assert executed == []
    assert model.calls[1][1] == []
    blocked_messages = [message for message in model.calls[1][0] if message.role == "tool"]
    assert [message.tool_call_id for message in blocked_messages] == ["call-1", "call-2"]
    assert all(json.loads(message.content)["reason"] == "agent_tool_budget_exhausted" for message in blocked_messages)
    events = await trace_store.list_by_run("tenant", "run-react")
    assert sum(event.event_type == RolloutEventType.AGENT_TOOL_BUDGET_EXHAUSTED for event in events) == 1
    assert sum(event.event_type == RolloutEventType.TOOL_CALL_BLOCKED for event in events) == 2
    assert not any(event.event_type == RolloutEventType.TOOL_CALL_STARTED for event in events)


@pytest.mark.asyncio
async def test_round_budget_blocks_next_batch_and_forces_final_call() -> None:
    executed: list[str] = []

    def handler(_: EmptyArgs) -> dict[str, Any]:
        executed.append("ran")
        return {"success": True}

    graph, model, trace_store = _graph(
        [
            ModelResponse(tool_calls=[ModelToolCall(id="allowed", name="toolA")], content=""),
            ModelResponse(tool_calls=[ModelToolCall(id="blocked", name="toolA")], content=""),
            ModelResponse(content="round final"),
        ],
        [_tool("toolA", handler)],
        max_rounds=1,
    )

    assert await graph.run(_run_context(), [ModelMessage(role="user", content="go")]) == "round final"
    assert executed == ["ran"]
    assert model.calls[-1][1] == []
    events = await trace_store.list_by_run("tenant", "run-react")
    budget = next(
        event for event in events if event.event_type == RolloutEventType.AGENT_TOOL_BUDGET_EXHAUSTED
    )
    assert budget.payload["reason"] == "agent_tool_round_budget_exhausted"
    assert budget.payload["toolRoundCount"] == 2


@pytest.mark.asyncio
async def test_final_call_tool_request_uses_nonempty_fallback_without_execution() -> None:
    executed: list[str] = []

    def handler(_: EmptyArgs) -> dict[str, Any]:
        executed.append("ran")
        return {"success": True}

    graph, model, trace_store = _graph(
        [
            ModelResponse(
                tool_calls=[
                    ModelToolCall(id="one", name="toolA"),
                    ModelToolCall(id="two", name="toolA"),
                ],
                content="",
            ),
            ModelResponse(
                content="partial answer",
                tool_calls=[ModelToolCall(id="forbidden-final", name="toolA")],
            ),
        ],
        [_tool("toolA", handler)],
        max_calls=1,
    )

    answer = await graph.run(_run_context(), [ModelMessage(role="user", content="go")])

    assert executed == []
    assert answer.startswith("partial answer")
    assert "未执行" in answer
    assert model.calls[-1][1] == []
    events = await trace_store.list_by_run("tenant", "run-react")
    assert RolloutEventType.AGENT_FINALIZATION_FALLBACK in [event.event_type for event in events]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_calls,reason",
    [
        ([ModelToolCall(id="", name="toolA")], "blank_tool_call_id"),
        ([ModelToolCall(id="call", name="")], "blank_tool_name"),
        (
            [ModelToolCall(id="same", name="toolA"), ModelToolCall(id="same", name="toolA")],
            "duplicate_tool_call_id_in_batch",
        ),
    ],
)
async def test_invalid_tool_protocol_rejects_batch_before_assistant_message(
    tool_calls: list[ModelToolCall],
    reason: str,
) -> None:
    executed: list[str] = []
    graph, model, trace_store = _graph(
        [ModelResponse(tool_calls=tool_calls, content=""), ModelResponse(content="safe final")],
        [_tool("toolA", lambda _: executed.append("ran") or {"success": True})],
    )

    assert await graph.run(_run_context(), [ModelMessage(role="user", content="go")]) == "safe final"
    assert executed == []
    final_messages = model.calls[1][0]
    assert not any(message.role == "assistant" and message.tool_calls for message in final_messages)
    assert final_messages[-1].role == "system"
    assert model.calls[1][1] == []
    events = await trace_store.list_by_run("tenant", "run-react")
    protocol_event = next(
        event for event in events if event.event_type == RolloutEventType.AGENT_MODEL_PROTOCOL_ERROR
    )
    assert reason in {item["reason"] for item in protocol_event.payload["violations"]}


@pytest.mark.asyncio
async def test_cross_round_duplicate_tool_id_rejects_second_round() -> None:
    executed: list[str] = []
    graph, model, trace_store = _graph(
        [
            ModelResponse(tool_calls=[ModelToolCall(id="repeat", name="toolA")], content=""),
            ModelResponse(tool_calls=[ModelToolCall(id="repeat", name="toolA")], content=""),
            ModelResponse(content="duplicate handled"),
        ],
        [_tool("toolA", lambda _: executed.append("ran") or {"success": True})],
    )

    assert await graph.run(_run_context(), [ModelMessage(role="user", content="go")]) == "duplicate handled"
    assert executed == ["ran"]
    assert model.calls[-1][1] == []
    events = await trace_store.list_by_run("tenant", "run-react")
    protocol = next(
        event for event in events if event.event_type == RolloutEventType.AGENT_MODEL_PROTOCOL_ERROR
    )
    assert protocol.payload["violations"] == [
        {"index": 0, "reason": "reused_tool_call_id_in_run"}
    ]


@pytest.mark.asyncio
async def test_empty_model_answer_returns_nonempty_fallback() -> None:
    graph, _, trace_store = _graph([ModelResponse(content="")], [])

    answer = await graph.run(_run_context(), [ModelMessage(role="user", content="go")])

    assert answer.strip()
    events = await trace_store.list_by_run("tenant", "run-react")
    assert events[-1].event_type == RolloutEventType.AGENT_FINALIZATION_FALLBACK


@pytest.mark.asyncio
async def test_reducer_preserves_multi_result_pairing_and_distinct_raw_refs() -> None:
    large = "x" * 5000
    reducer = ToolResultReducer(
        budget=ContextBudget(
            tool_result_compress_threshold_tokens=10,
            tool_results_to_keep=3,
        ),
        estimator=ApproxTokenEstimator(),
        compression_backend=DeterministicContentCompressionBackend(),
    )
    graph, model, _ = _graph(
        [
            ModelResponse(
                tool_calls=[
                    ModelToolCall(id="large-a", name="toolA"),
                    ModelToolCall(id="large-b", name="toolB"),
                ],
                content="",
            ),
            ModelResponse(content="done"),
        ],
        [
            _tool("toolA", lambda _: {"success": True, "data": large}),
            _tool("toolB", lambda _: {"success": True, "data": large}),
        ],
        reducer=reducer,
    )

    assert await graph.run(_run_context(), [ModelMessage(role="user", content="go")]) == "done"
    tool_messages = [message for message in model.calls[1][0] if message.role == "tool"]
    payloads = [json.loads(message.content) for message in tool_messages]
    assert [message.tool_call_id for message in tool_messages] == ["large-a", "large-b"]
    assert len({payload["raw_ref"] for payload in payloads}) == 2


class SelectiveFailTraceStore(InMemoryRolloutEventStore):
    def __init__(self, fail_on: RolloutEventType | None = None) -> None:
        super().__init__()
        self.fail_on = fail_on

    async def append_event(self, run_context, event_type, payload=None, **kwargs):
        if event_type == self.fail_on:
            raise RuntimeError(f"trace failed: {event_type}")
        return await super().append_event(run_context, event_type, payload, **kwargs)


@pytest.mark.asyncio
async def test_runtime_start_complete_and_fail_trace_errors_cleanup_state() -> None:
    service = AgentHarnessService.placeholder()
    runtime = service.runtime
    context = AgentRequestContext("tenant", "user", "agent", "cleanup")

    start_store = SelectiveFailTraceStore(RolloutEventType.RUN_STARTED)
    runtime.trace_store = start_store
    with pytest.raises(RuntimeError, match="RUN_STARTED"):
        await runtime.start_run(context, "start")
    assert runtime._active_history_by_run == {}
    assert runtime._history_events_by_run == {}

    for terminal_event, method_name in [
        (RolloutEventType.RUN_COMPLETED, "complete_run"),
        (RolloutEventType.RUN_FAILED, "fail_run"),
    ]:
        store = SelectiveFailTraceStore()
        runtime.trace_store = store
        run_context = await runtime.start_run(context, "terminal")
        store.fail_on = terminal_event
        with pytest.raises(RuntimeError, match=terminal_event.value):
            await getattr(runtime, method_name)(run_context, "value")
        assert run_context.run_id not in runtime._active_history_by_run
        assert run_context.run_id not in runtime._history_events_by_run


@pytest.mark.asyncio
async def test_chat_stream_cancellation_cleans_runtime_and_graph_state() -> None:
    started = asyncio.Event()

    class CancellableGraph:
        def __init__(self) -> None:
            self.cleaned: list[str] = []

        async def run_stream(
            self,
            run_context: RunContext,
            messages: list[ModelMessage],
        ) -> AsyncIterator[AgentStreamEvent]:
            _ = run_context, messages
            started.set()
            await asyncio.Event().wait()
            yield AgentStreamEvent(type="content", data="unreachable")

        def cleanup_run(self, run_id: str) -> None:
            self.cleaned.append(run_id)

    service = AgentHarnessService.placeholder()
    graph = CancellableGraph()
    service.graph = graph  # type: ignore[assignment]
    context = AgentRequestContext("tenant", "user", "agent", "cancel")
    stream = service.chat_stream(context, "wait")
    task = asyncio.create_task(anext(stream))
    await started.wait()
    active_run_id = next(iter(service.runtime._active_history_by_run))

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await stream.aclose()

    assert active_run_id not in service.runtime._active_history_by_run
    assert active_run_id not in service.runtime._history_events_by_run
    assert service.memory_runtime is not None
    assert not service.memory_runtime.core_version_snapshots.contains(active_run_id)
    assert graph.cleaned == [active_run_id]


def test_harness_budget_settings_validate_at_least_one() -> None:
    settings = Settings(_env_file=None)
    assert settings.agent_max_tool_rounds == 4
    assert settings.agent_max_tool_calls_per_run == 8
    with pytest.raises(ValueError):
        Settings(agent_max_tool_rounds=0, _env_file=None)
    with pytest.raises(ValueError):
        Settings(agent_max_tool_calls_per_run=0, _env_file=None)
