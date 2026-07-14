from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.graph import SkeletonAgentGraph
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore
from superbiz_agent.model_gateway.base import ModelMessage, ModelResponse, ModelToolCall
from superbiz_agent.model_gateway.errors import (
    ModelGatewayContextOverflowError,
    ModelGatewayProviderError,
    ModelGatewayTimeoutError,
)
from superbiz_agent.model_gateway.factory import build_model_gateway
from superbiz_agent.model_gateway.openai_compatible import OpenAICompatibleModelGateway
from superbiz_agent.model_gateway.stub import StubModelGateway
from superbiz_agent.tools.builtin import build_datetime_tool
from superbiz_agent.tools.gateway import ToolGateway
from superbiz_agent.tools.registry import ToolDefinition, ToolRegistry


class FakeCompletions:
    def __init__(self, response: Any | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.last_request: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeOpenAIClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class RecordingGateway:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.seen_tools: list[ToolDefinition] | None = None

    async def complete(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        self.seen_tools = tools
        if self.error is not None:
            raise self.error
        return ModelResponse(content="ok")


def _response(
    *,
    content: str | None = "answer",
    tool_calls: list[Any] | None = None,
    usage: Any | None = None,
    finish_reason: str = "stop",
) -> Any:
    return SimpleNamespace(
        id="response-1",
        model="qwen-plus",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or []),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def test_model_gateway_factory_defaults_to_stub() -> None:
    gateway = build_model_gateway(Settings(model_provider="stub", _env_file=None))

    assert isinstance(gateway, StubModelGateway)


def test_model_gateway_factory_builds_openai_compatible_provider() -> None:
    gateway = build_model_gateway(
        Settings(
            model_provider="qwen-openai-compatible",
            model_api_key="fake-key",
            model_base_url="https://example.invalid/v1",
            _env_file=None,
        )
    )

    assert isinstance(gateway, OpenAICompatibleModelGateway)
    assert gateway.provider_name == "qwen-openai-compatible"


def test_model_gateway_missing_api_key_fails_for_real_provider() -> None:
    with pytest.raises(ValueError, match="model_api_key"):
        build_model_gateway(
            Settings(
                model_provider="openai-compatible",
                model_api_key=None,
                _env_file=None,
            )
        )


@pytest.mark.asyncio
async def test_openai_compatible_gateway_parses_content_usage_and_finish_reason() -> None:
    completions = FakeCompletions(
        _response(
            content="hello from model",
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
            finish_reason="stop",
        )
    )
    gateway = OpenAICompatibleModelGateway(
        model_name="qwen-plus",
        api_key="fake-key",
        timeout_ms=1_000,
        client=FakeOpenAIClient(completions),
    )

    response = await gateway.complete(
        [ModelMessage(role="user", content="hello")],
        tools=[build_datetime_tool()],
    )

    assert response.content == "hello from model"
    assert response.tool_calls == []
    assert response.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    assert response.finish_reason == "stop"
    assert response.raw["usage"] == response.usage
    assert response.raw["finish_reason"] == "stop"

    assert completions.last_request is not None
    assert completions.last_request["model"] == "qwen-plus"
    assert completions.last_request["messages"] == [{"role": "user", "content": "hello"}]
    assert completions.last_request["tool_choice"] == "auto"
    openai_tool = completions.last_request["tools"][0]
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "getCurrentDateTime"
    assert "timezone" in openai_tool["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_openai_compatible_gateway_parses_tool_call_arguments() -> None:
    completions = FakeCompletions(
        _response(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="queryLogs",
                        arguments=(
                            '{"region":"ap-guangzhou","logTopic":"application-logs",'
                            '"query":"level:ERROR","limit":20}'
                        ),
                    ),
                )
            ],
            finish_reason="tool_calls",
        )
    )
    gateway = OpenAICompatibleModelGateway(
        model_name="qwen-plus",
        api_key="fake-key",
        client=FakeOpenAIClient(completions),
    )

    response = await gateway.complete([ModelMessage(role="user", content="查日志")])

    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [
        ModelToolCall(
            id="call-1",
            name="queryLogs",
            arguments={
                "region": "ap-guangzhou",
                "logTopic": "application-logs",
                "query": "level:ERROR",
                "limit": 20,
            },
        )
    ]
    assert response.raw["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_openai_compatible_gateway_keeps_invalid_tool_arguments_empty() -> None:
    completions = FakeCompletions(
        _response(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call-invalid",
                    function=SimpleNamespace(
                        name="queryLogs",
                        arguments="{not valid json",
                    ),
                )
            ],
            finish_reason="tool_calls",
        )
    )
    gateway = OpenAICompatibleModelGateway(
        model_name="qwen-plus",
        api_key="fake-key",
        client=FakeOpenAIClient(completions),
    )

    response = await gateway.complete([ModelMessage(role="user", content="查日志")])

    assert response.tool_calls == [
        ModelToolCall(id="call-invalid", name="queryLogs", arguments={})
    ]
    assert response.raw["tool_call_parse_errors"] == [
        {"index": "0", "reason": "invalid_json_arguments"}
    ]


@pytest.mark.asyncio
async def test_openai_compatible_gateway_preserves_blank_tool_id_and_name() -> None:
    completions = FakeCompletions(
        _response(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="",
                    function=SimpleNamespace(name="queryLogs", arguments="{}"),
                ),
                SimpleNamespace(
                    id="call-blank-name",
                    function=SimpleNamespace(name="", arguments="{}"),
                ),
            ],
            finish_reason="tool_calls",
        )
    )
    gateway = OpenAICompatibleModelGateway(
        model_name="qwen-plus",
        api_key="fake-key",
        client=FakeOpenAIClient(completions),
    )

    response = await gateway.complete([ModelMessage(role="user", content="查日志")])

    assert response.tool_calls == [
        ModelToolCall(id="", name="queryLogs", arguments={}),
        ModelToolCall(id="call-blank-name", name="", arguments={}),
    ]
    assert response.raw["tool_call_parse_errors"] == [
        {"index": "0", "reason": "missing_tool_call_id"},
        {"index": "1", "reason": "invalid_tool_name"},
    ]


@pytest.mark.asyncio
async def test_openai_compatible_gateway_maps_timeout_error() -> None:
    gateway = OpenAICompatibleModelGateway(
        model_name="qwen-plus",
        api_key="fake-key",
        client=FakeOpenAIClient(FakeCompletions(error=TimeoutError("request timed out"))),
    )

    with pytest.raises(ModelGatewayTimeoutError):
        await gateway.complete([ModelMessage(role="user", content="hello")])


@pytest.mark.asyncio
async def test_openai_compatible_gateway_maps_context_overflow_error() -> None:
    gateway = OpenAICompatibleModelGateway(
        model_name="qwen-plus",
        api_key="fake-key",
        client=FakeOpenAIClient(
            FakeCompletions(error=Exception("maximum context length exceeded"))
        ),
    )

    with pytest.raises(ModelGatewayContextOverflowError):
        await gateway.complete([ModelMessage(role="user", content="hello")])


@pytest.mark.asyncio
async def test_graph_records_model_timeout_and_keeps_failed_trace() -> None:
    trace_store = InMemoryRolloutEventStore()
    registry = ToolRegistry()
    registry.register(build_datetime_tool())
    gateway = RecordingGateway(ModelGatewayTimeoutError("provider timed out"))
    graph = SkeletonAgentGraph(
        model_gateway=gateway,
        tool_gateway=ToolGateway(registry, trace_store),
        trace_store=trace_store,
    )
    run_context = _run_context("run-timeout")

    with pytest.raises(ModelGatewayTimeoutError):
        await graph.run(run_context, [ModelMessage(role="user", content="hello")])

    assert gateway.seen_tools is not None
    assert [tool.name for tool in gateway.seen_tools] == ["getCurrentDateTime"]
    events = await trace_store.list_by_run("tenant-a", "run-timeout")
    assert [event.event_type for event in events] == [
        RolloutEventType.MODEL_CALL_STARTED,
        RolloutEventType.MODEL_CALL_TIMEOUT,
        RolloutEventType.MODEL_CALL_FAILED,
    ]
    assert events[-1].payload["errorType"] == "timeout"


@pytest.mark.asyncio
async def test_graph_records_model_provider_failure() -> None:
    trace_store = InMemoryRolloutEventStore()
    gateway = RecordingGateway(
        ModelGatewayProviderError(
            "provider failed",
            provider="openai-compatible",
            status_code=502,
            retryable=True,
        )
    )
    graph = SkeletonAgentGraph(
        model_gateway=gateway,
        tool_gateway=ToolGateway(ToolRegistry(), trace_store),
        trace_store=trace_store,
    )
    run_context = _run_context("run-provider-error")

    with pytest.raises(ModelGatewayProviderError):
        await graph.run(run_context, [ModelMessage(role="user", content="hello")])

    events = await trace_store.list_by_run("tenant-a", "run-provider-error")
    assert [event.event_type for event in events] == [
        RolloutEventType.MODEL_CALL_STARTED,
        RolloutEventType.MODEL_CALL_FAILED,
    ]
    assert events[-1].payload["errorType"] == "provider-error"
    assert events[-1].payload["statusCode"] == 502
    assert events[-1].payload["retryable"] is True


@pytest.mark.asyncio
async def test_graph_records_model_context_overflow_and_keeps_failed_trace() -> None:
    trace_store = InMemoryRolloutEventStore()
    gateway = RecordingGateway(ModelGatewayContextOverflowError("maximum context length exceeded"))
    graph = SkeletonAgentGraph(
        model_gateway=gateway,
        tool_gateway=ToolGateway(ToolRegistry(), trace_store),
        trace_store=trace_store,
    )
    run_context = _run_context("run-context-overflow")

    with pytest.raises(ModelGatewayContextOverflowError):
        await graph.run(run_context, [ModelMessage(role="user", content="hello")])

    events = await trace_store.list_by_run("tenant-a", "run-context-overflow")
    assert [event.event_type for event in events] == [
        RolloutEventType.MODEL_CALL_STARTED,
        RolloutEventType.MODEL_CALL_CONTEXT_OVERFLOW,
        RolloutEventType.MODEL_CALL_FAILED,
    ]
    assert events[-1].payload["errorType"] == "context-overflow"


def _run_context(run_id: str) -> RunContext:
    return RunContext(
        request_context=AgentRequestContext("tenant-a", "user-a", "agent-a", "session-a"),
        run_id=run_id,
        prompt_version="ops-agent-system-v2",
        tool_schema_version="ops-tools-v1",
        model_provider="openai-compatible",
    )
