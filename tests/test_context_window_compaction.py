from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import BaseModel

from superbiz_agent.config import Settings
from superbiz_agent.harness.content_compression import (
    ContentCompressionResult,
    DeterministicContentCompressionBackend,
    HeadroomContentCompressionBackend,
)
from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.context_budget import ContextBudget
from superbiz_agent.harness.events import RolloutEvent, RolloutEventType
from superbiz_agent.harness.history import recover_active_history
from superbiz_agent.harness.graph import SkeletonAgentGraph
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore
from superbiz_agent.harness.token_estimator import ApproxTokenEstimator
from superbiz_agent.harness.tool_result_reducer import ToolResultReducer
from superbiz_agent.model_gateway.base import ModelMessage, ModelResponse, ModelToolCall
from superbiz_agent.security.permissions import LOCAL_DEFAULT_PERMISSIONS
from superbiz_agent.tools.gateway import ToolGateway
from superbiz_agent.tools.policies import ToolPolicy
from superbiz_agent.tools.registry import ToolDefinition, ToolRegistry


class RecordingCompressionBackend:
    backend_name = "recording"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def compress(self, content: str, *, metadata: dict | None = None) -> ContentCompressionResult:
        self.calls.append(content)
        return ContentCompressionResult(
            content=f"compressed::{len(content)}",
            backend=self.backend_name,
            compressed=True,
        )


def _run_context() -> RunContext:
    return RunContext(
        request_context=AgentRequestContext(
            "tenant",
            "user",
            "agent",
            "session",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        ),
        run_id="run-context",
        prompt_version="ops-agent-system-v2",
        tool_schema_version="ops-tools-v1",
        model_provider="stub",
    )


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


def test_approx_token_estimator_counts_text_and_tool_messages() -> None:
    estimator = ApproxTokenEstimator()
    english = estimator.estimate_text("abcd" * 10)
    chinese = estimator.estimate_text("中文" * 10)
    messages = estimator.estimate_messages(
        [
            ModelMessage(role="user", content="hello"),
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=[ModelToolCall(id="call-1", name="queryLogs", arguments={"limit": 2})],
            ),
            ModelMessage(role="tool", name="queryLogs", tool_call_id="call-1", content="{}"),
        ]
    )

    assert english >= 10
    assert chinese >= 10
    assert messages > english


def test_context_budget_derives_trigger_and_target_tokens() -> None:
    budget = ContextBudget(
        max_input_tokens=1000,
        reserved_output_tokens=100,
        compaction_trigger_ratio=0.8,
        compaction_target_ratio=0.5,
    )

    assert budget.effective_input_budget == 900
    assert budget.trigger_tokens == 720
    assert budget.target_tokens_after_compaction == 450
    assert budget.should_compact(721) is True
    assert budget.should_compact(720) is False


def test_tool_result_reducer_redacts_and_does_not_compress_under_threshold() -> None:
    estimator = ApproxTokenEstimator()
    backend = RecordingCompressionBackend()
    reducer = ToolResultReducer(
        budget=ContextBudget(tool_result_compress_threshold_tokens=1000, tool_results_to_keep=3),
        estimator=estimator,
        compression_backend=backend,
    )
    messages = [
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=[ModelToolCall(id="call-small", name="queryLogs", arguments={})],
        ),
        ModelMessage(
            role="tool",
            name="queryLogs",
            tool_call_id="call-small",
            content=json.dumps(
                {"success": True, "api_key": "secret-value", "logs": [{"message": "ok"}]},
                ensure_ascii=False,
            ),
            metadata={"rollout_sequence": 42},
        ),
    ]

    result = reducer.reduce(messages, run_context=_run_context())
    payload = json.loads(result.messages[1].content)

    assert backend.calls == []
    assert result.raw_refs[0].endswith("/sequence/42")
    assert "compression" not in payload
    assert payload["success"] is True
    assert payload["api_key"] == "[REDACTED]"
    assert result.messages[0].tool_calls[0].id == result.messages[1].tool_call_id


def test_tool_result_reducer_compresses_large_output_and_omits_orphan_results() -> None:
    estimator = ApproxTokenEstimator()
    backend = RecordingCompressionBackend()
    reducer = ToolResultReducer(
        budget=ContextBudget(tool_result_compress_threshold_tokens=1000, tool_results_to_keep=3),
        estimator=estimator,
        compression_backend=backend,
    )
    large_rows = [{"line": index, "message": "x" * 120} for index in range(80)]
    messages = [
        ModelMessage(role="tool", name="queryLogs", tool_call_id="orphan", content="{}"),
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=[ModelToolCall(id="call-large", name="queryLogs", arguments={})],
        ),
        ModelMessage(
            role="tool",
            name="queryLogs",
            tool_call_id="call-large",
            content=json.dumps({"success": True, "logs": large_rows}, ensure_ascii=False),
            metadata={"rollout_sequence": 77},
        ),
    ]

    result = reducer.reduce(messages, run_context=_run_context())
    payload = json.loads(result.messages[-1].content)

    assert len(backend.calls) == 1
    assert result.orphan_tool_result_count == 1
    assert result.compressed_tool_result_count == 1
    assert payload["compression"]["backend"] == "recording"
    assert payload["compression"]["beforeTokens"] > 1000
    assert payload["summary"].startswith("compressed::")
    assert payload["raw_ref"].endswith("/sequence/77")
    assert result.messages[-2].tool_calls[0].id == result.messages[-1].tool_call_id


def test_headroom_backend_degrades_without_required_package() -> None:
    backend = HeadroomContentCompressionBackend(
        fallback=DeterministicContentCompressionBackend(max_chars=600)
    )

    result = backend.compress("x" * 2000)

    assert result.backend.startswith("headroom:") or result.backend == "headroom"
    if result.backend.startswith("headroom:"):
        assert result.warnings


def test_recover_active_history_uses_latest_trimmed_boundary() -> None:
    events = [
        _event(RolloutEventType.USER_MESSAGE_APPENDED, {"content": "old user"}, sequence=1),
        _event(RolloutEventType.ASSISTANT_MESSAGE_APPENDED, {"content": "old assistant"}, sequence=2),
        _event(
            RolloutEventType.HISTORY_TRIMMED,
            {
                "summary": "<conversation_summary>summary</conversation_summary>",
                "toSequence": 2,
                "status": "success",
            },
            sequence=3,
        ),
        _event(RolloutEventType.USER_MESSAGE_APPENDED, {"content": "recent user"}, sequence=4),
        _event(
            RolloutEventType.ASSISTANT_MESSAGE_APPENDED,
            {"content": "recent assistant"},
            sequence=5,
        ),
    ]

    history = recover_active_history(events)

    assert [message.role for message in history] == ["system", "user", "assistant"]
    assert history[0].content.startswith("<conversation_summary>")
    assert [message.content for message in history[1:]] == ["recent user", "recent assistant"]


@pytest.mark.asyncio
async def test_graph_reduces_current_run_large_tool_result_before_second_model_call() -> None:
    class EmptyArgs(BaseModel):
        pass

    class TwoStepModelGateway:
        def __init__(self) -> None:
            self.calls: list[list[ModelMessage]] = []

        async def complete(self, messages, tools=None) -> ModelResponse:
            self.calls.append(messages)
            if len(self.calls) == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ModelToolCall(id="call-current-large", name="largeTool", arguments={})
                    ],
                )
            return ModelResponse(content="done")

    def large_tool_handler(args: EmptyArgs) -> dict:
        _ = args
        return {
            "success": True,
            "logs": [{"line": index, "message": "x" * 120} for index in range(80)],
        }

    estimator = ApproxTokenEstimator()
    backend = RecordingCompressionBackend()
    reducer = ToolResultReducer(
        budget=ContextBudget(tool_result_compress_threshold_tokens=1000),
        estimator=estimator,
        compression_backend=backend,
    )
    trace_store = InMemoryRolloutEventStore()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="largeTool",
            description="Return a large tool result.",
            args_model=EmptyArgs,
            handler=large_tool_handler,
            policy=ToolPolicy(tool_name="largeTool"),
        )
    )
    model_gateway = TwoStepModelGateway()
    graph = SkeletonAgentGraph(
        model_gateway=model_gateway,
        tool_gateway=ToolGateway(registry, trace_store),
        trace_store=trace_store,
        tool_result_reducer=reducer,
    )
    run_context = _run_context()

    answer = await graph.run(run_context, [ModelMessage(role="user", content="run large tool")])
    events = await trace_store.list_by_run("tenant", run_context.run_id)
    second_call_tool_messages = [
        message for message in model_gateway.calls[1] if message.role == "tool"
    ]
    tool_payload = json.loads(second_call_tool_messages[0].content)

    assert answer == "done"
    assert len(backend.calls) == 1
    assert tool_payload["summary"].startswith("compressed::")
    assert tool_payload["raw_ref"].endswith("/tool_call/call-current-large/sequence/4")
    assert any(
        event.event_type == RolloutEventType.MODEL_CALL_STARTED
        and event.payload.get("toolResultReduction", {}).get("compressedToolResultCount") == 1
        for event in events
    )


@pytest.mark.asyncio
async def test_runtime_compacts_long_history_and_records_budget_trace() -> None:
    service = AgentHarnessService.build_default(
        Settings(
            memory_enabled=False,
            context_max_tokens=900,
            context_reserved_output_tokens=0,
            context_compaction_trigger_ratio=0.35,
            context_compaction_target_ratio=0.25,
            context_recent_turns_to_keep=1,
            context_content_compression_backend="deterministic",
            _env_file=None,
        )
    )
    context = AgentRequestContext("tc", "uc", "ac", "compact-session")
    long_text = "请分析这个历史约束，必须保留用户要求。 " + ("x" * 2000)

    await service.chat(context, long_text)
    await service.chat(context, f"{long_text} second")
    result = await service.chat(context, "继续处理当前问题")
    events = await service.trace_store.list_by_session(context)
    trimmed = [event for event in events if event.event_type == RolloutEventType.HISTORY_TRIMMED]
    context_events = [
        event for event in events if event.event_type == RolloutEventType.CONTEXT_ASSEMBLED
    ]

    assert result.success is True
    assert trimmed
    assert trimmed[-1].payload["summary"].startswith("<conversation_summary>")
    assert trimmed[-1].payload["toSequence"] < trimmed[-1].sequence
    latest_context_payload = context_events[-1].payload
    assert latest_context_payload["compactionTriggered"] is True
    assert latest_context_payload["estimatedInputTokens"] > 0
    assert latest_context_payload["tokenBudget"]["effectiveInputBudgetTokens"] == 900
    assert {usage["name"] for usage in latest_context_payload["componentUsage"]} >= {
        "system_prompt",
        "active_history",
        "current_user_message",
    }

    recovered = recover_active_history(events)
    assert recovered[0].role == "system"
    assert recovered[0].content.startswith("<conversation_summary>")
    assert not any(message.content == "old user" for message in recovered)
