import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from superbiz_agent.api.app import create_app
from superbiz_agent.config import Settings
from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.context_assembler import ContextAssembler
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.model_gateway.base import ModelMessage, ModelToolCall
from superbiz_agent.model_gateway.stub import StubModelGateway
from superbiz_agent.security.permissions import LOCAL_DEFAULT_PERMISSIONS
from superbiz_agent.prompts.registry import PromptRegistry
from superbiz_agent.tools.builtin import build_datetime_tool
from superbiz_agent.tools.errors import ToolErrorType
from superbiz_agent.tools.gateway import ToolGateway
from superbiz_agent.tools.policies import ToolAction
from superbiz_agent.tools.registry import ToolRegistry


def test_app_imports_and_chat_time_question_runs_skeleton_loop() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"Id": "s1", "Question": "现在几点？"},
        headers={
            "X-Tenant-Id": "tenant-a",
            "X-User-Id": "user-a",
            "X-Agent-Id": "agent-a",
            "X-Request-Id": "request-a",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["success"] is True
    assert "Asia/Shanghai" in body["data"]["answer"]

    service = app.state.harness_service
    events = asyncio.run(
        service.trace_store.list_by_session(
            AgentRequestContext("tenant-a", "user-a", "agent-a", "s1")
        )
    )
    event_types = [event.event_type for event in events]
    assert event_types == [
        RolloutEventType.RUN_STARTED,
        RolloutEventType.USER_MESSAGE_APPENDED,
        RolloutEventType.CONTEXT_ASSEMBLED,
        RolloutEventType.MEMORY_INJECTED,
        RolloutEventType.MODEL_CALL_STARTED,
        RolloutEventType.MODEL_CALL_COMPLETED,
        RolloutEventType.TOOL_CALL_STARTED,
        RolloutEventType.TOOL_CALL_COMPLETED,
        RolloutEventType.MODEL_CALL_STARTED,
        RolloutEventType.MODEL_CALL_COMPLETED,
        RolloutEventType.ASSISTANT_MESSAGE_APPENDED,
        RolloutEventType.RUN_COMPLETED,
    ]
    assert all(event.tenant_id == "tenant-a" for event in events)
    assert all(event.user_id == "user-a" for event in events)
    assert all(event.agent_id == "agent-a" for event in events)
    assert all(event.session_id == "s1" for event in events)
    assert all(event.run_id for event in events)
    assert all(event.event_id for event in events)
    assert all(event.occurred_at for event in events)
    assert all(isinstance(event.payload, dict) for event in events)
    assert [event.sequence for event in events] == sorted(event.sequence for event in events)


def test_chat_empty_question_returns_java_compatible_error() -> None:
    client = TestClient(create_app())
    response = client.post("/api/chat", json={"Id": "s1", "Question": "  "})

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"] == {
        "success": False,
        "answer": None,
        "errorMessage": "问题内容不能为空",
    }


def test_request_context_defaults_and_conversation_key() -> None:
    context = AgentRequestContext(None, None, None, "session-a")

    assert context.tenant_id == "default-tenant"
    assert context.user_id == "default-user"
    assert context.agent_id == "ops-agent"
    assert context.session_id == "session-a"
    assert context.conversation_key == "default-tenant:default-user:ops-agent:session-a"


def test_rollout_event_type_contains_memory_events() -> None:
    assert RolloutEventType.CORE_MEMORY_UPDATED == "CORE_MEMORY_UPDATED"
    assert RolloutEventType.CORE_MEMORY_UNCHANGED == "CORE_MEMORY_UNCHANGED"
    assert RolloutEventType.ARCHIVAL_MEMORY_WRITTEN == "ARCHIVAL_MEMORY_WRITTEN"


def test_prompt_registry_loads_versioned_prompt() -> None:
    registry = PromptRegistry(Path(__file__).resolve().parents[1] / "prompts")
    prompt_v2 = registry.load("ops-agent-system-v2")
    prompt_v3 = registry.load("ops-agent-system-v3")

    assert "SuperBizAgent" in prompt_v2
    assert "长期记忆规则" in prompt_v2
    assert "两道门槛" in prompt_v3
    assert "不得重复询问" in prompt_v3
    assert "unchanged" in prompt_v3
    assert "any-match" in prompt_v3
    assert "才使用 memory 工具" not in prompt_v3
    assert "按长期记忆检索规则使用 searchMemory" in prompt_v3
    assert "明确持久意图或持久语义" in prompt_v3
    assert "按长期记忆写入决策使用 updateCoreMemory" in prompt_v3


def test_default_prompt_and_tool_schema_versions_are_current_versions() -> None:
    settings = Settings(_env_file=None)

    assert settings.prompt_version == "ops-agent-system-v3"
    assert settings.tool_schema_version == "ops-tools-v3"


@pytest.mark.asyncio
async def test_context_assembler_orders_system_history_user_and_payload() -> None:
    registry = PromptRegistry(Path(__file__).resolve().parents[1] / "prompts")
    assembler = ContextAssembler(registry)
    assembled = await assembler.assemble(
        prompt_version="ops-agent-system-v2",
        active_history=[ModelMessage(role="assistant", content="历史回答")],
        current_user_message="当前问题",
    )

    assert [message.role for message in assembled.messages] == ["system", "assistant", "user"]
    assert assembled.messages[-1].content == "当前问题"
    assert assembled.trace_payload["promptVersion"] == "ops-agent-system-v2"
    assert assembled.trace_payload["messageCount"] == 3
    assert assembled.trace_payload["estimatedChars"] > len("当前问题")
    assert assembled.trace_payload["hasCoreMemory"] is False
    assert assembled.trace_payload["hasMemoryIndex"] is False
    assert assembled.trace_payload["hasMemoryMetadata"] is False


@pytest.mark.asyncio
async def test_stub_model_gateway_datetime_tool_call_and_final_answer() -> None:
    gateway = StubModelGateway()
    first = await gateway.complete([ModelMessage(role="user", content="date please")])

    assert first.content == ""
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].name == "getCurrentDateTime"
    assert first.tool_calls[0].arguments == {"timezone": "Asia/Shanghai"}

    final = await gateway.complete(
        [
            ModelMessage(role="user", content="date please"),
            ModelMessage(
                role="tool",
                name="getCurrentDateTime",
                content=(
                    '{"success": true, "data": '
                    '{"timezone": "Asia/Shanghai", "isoTime": "2026-07-05T12:00:00+08:00"}}'
                ),
            ),
        ]
    )

    assert final.tool_calls == []
    assert "Asia/Shanghai" in final.content


@pytest.mark.asyncio
async def test_stub_model_gateway_non_datetime_question_returns_stub_answer() -> None:
    gateway = StubModelGateway()
    response = await gateway.complete([ModelMessage(role="user", content="hello")])

    assert response.tool_calls == []
    assert response.content == "[stub] hello"


@pytest.mark.asyncio
async def test_tool_gateway_executes_datetime_tool_and_records_trace() -> None:
    service = AgentHarnessService.placeholder()
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "t1",
            "u1",
            "a1",
            "s-tool",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    gateway = service.graph.tool_gateway

    result = await gateway.execute(
        run_context,
        ModelToolCall(
            id="call-1",
            name="getCurrentDateTime",
            arguments={"timezone": "Asia/Shanghai"},
        ),
    )

    assert result.result["success"] is True
    assert result.result["data"]["timezone"] == "Asia/Shanghai"
    events = await service.trace_store.list_by_run("t1", run_context.run_id)
    assert [event.event_type for event in events[-2:]] == [
        RolloutEventType.TOOL_CALL_STARTED,
        RolloutEventType.TOOL_CALL_COMPLETED,
    ]
    started_payload = events[-2].payload
    assert started_payload["timeoutSeconds"] == 1
    assert started_payload["maxRetries"] == 0


@pytest.mark.asyncio
async def test_tool_gateway_unknown_tool_and_validation_errors_are_structured() -> None:
    service = AgentHarnessService.placeholder()
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "t2",
            "u2",
            "a2",
            "s-tool-errors",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    gateway = service.graph.tool_gateway

    unknown = await gateway.execute(
        run_context,
        ModelToolCall(id="call-unknown", name="missingTool", arguments={}),
    )
    invalid = await gateway.execute(
        run_context,
        ModelToolCall(id="call-invalid", name="getCurrentDateTime", arguments={"timezone": 123}),
    )

    assert unknown.result["success"] is False
    assert unknown.result["error_type"] == ToolErrorType.TOOL_BLOCKED.value
    assert invalid.result["success"] is False
    assert invalid.result["error_type"] == ToolErrorType.PARAM_VALIDATION_FAILED.value

    events = await service.trace_store.list_by_run("t2", run_context.run_id)
    assert RolloutEventType.TOOL_CALL_BLOCKED in [event.event_type for event in events]
    assert RolloutEventType.TOOL_CALL_FAILED in [event.event_type for event in events]


@pytest.mark.asyncio
async def test_tool_gateway_policy_deny_returns_blocked_error() -> None:
    service = AgentHarnessService.placeholder()
    definition = build_datetime_tool()
    denied_definition = type(definition)(
        name=definition.name,
        description=definition.description,
        args_model=definition.args_model,
        handler=definition.handler,
        policy=type(definition.policy)(
            tool_name=definition.name,
            danger_level=definition.policy.danger_level,
            action=ToolAction.DENY,
            timeout_seconds=definition.policy.timeout_seconds,
            max_retries=definition.policy.max_retries,
            idempotent=definition.policy.idempotent,
        ),
    )
    registry = ToolRegistry()
    registry.register(denied_definition)
    gateway = ToolGateway(registry, service.trace_store)
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "t4",
            "u4",
            "a4",
            "s-deny",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )

    denied = await gateway.execute(
        run_context,
        ModelToolCall(id="call-deny", name="getCurrentDateTime", arguments={}),
    )

    assert denied.result["success"] is False
    assert denied.result["error_type"] == ToolErrorType.TOOL_BLOCKED.value
    events = await service.trace_store.list_by_run("t4", run_context.run_id)
    assert events[-1].event_type == RolloutEventType.TOOL_CALL_BLOCKED
    assert events[-1].payload["reason"] == "policy_deny"


@pytest.mark.asyncio
async def test_harness_service_chat_returns_run_id_and_clear_removes_session_trace() -> None:
    service = AgentHarnessService.placeholder()
    context = AgentRequestContext(
        "t3",
        "u3",
        "a3",
        "s-clear",
        permissions=LOCAL_DEFAULT_PERMISSIONS,
        auth_mode="dev_headers",
    )
    result = await service.chat(context, "现在日期是什么？")

    assert result.success is True
    assert result.run_id
    events = await service.trace_store.list_by_session(context)
    assert RolloutEventType.RUN_COMPLETED in [event.event_type for event in events]

    await service.clear(context)
    assert await service.trace_store.list_by_session(context) == []
