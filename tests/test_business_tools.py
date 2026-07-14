from __future__ import annotations

import pytest

from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.model_gateway.base import ModelMessage, ModelToolCall
from superbiz_agent.model_gateway.stub import StubModelGateway
from superbiz_agent.security.permissions import LOCAL_DEFAULT_PERMISSIONS
from superbiz_agent.tools.builtin import build_builtin_tools
from superbiz_agent.tools.errors import ToolErrorType
from superbiz_agent.tools.policies import DangerLevel


def _tool_map():
    return {definition.name: definition for definition in build_builtin_tools()}


def test_business_tools_are_registered_with_expected_schema_and_policy() -> None:
    tools = _tool_map()

    assert set(tools) == {
        "getCurrentDateTime",
        "getAvailableLogTopics",
        "queryLogs",
        "queryPrometheusAlerts",
        "queryInternalDocs",
    }

    log_topics = tools["getAvailableLogTopics"]
    assert log_topics.args_model.model_json_schema()["additionalProperties"] is False
    assert log_topics.policy.danger_level == DangerLevel.LOW
    assert log_topics.policy.timeout_seconds == 2
    assert log_topics.policy.max_retries == 0
    assert log_topics.policy.idempotent is True

    alerts = tools["queryPrometheusAlerts"]
    assert alerts.args_model.model_json_schema()["additionalProperties"] is False
    assert alerts.policy.danger_level == DangerLevel.MEDIUM
    assert alerts.policy.timeout_seconds == 5
    assert alerts.policy.max_retries == 1
    assert alerts.policy.idempotent is True

    docs = tools["queryInternalDocs"]
    docs_schema = docs.args_model.model_json_schema()
    assert docs_schema["required"] == ["query"]
    assert docs_schema["properties"]["query"]["minLength"] == 1
    assert docs.policy.danger_level == DangerLevel.MEDIUM
    assert docs.policy.timeout_seconds == 8
    assert docs.policy.max_retries == 1
    assert docs.policy.idempotent is True

    logs = tools["queryLogs"]
    logs_schema = logs.args_model.model_json_schema()
    assert logs_schema["required"] == ["region", "logTopic"]
    assert logs_schema["properties"]["region"]["enum"] == [
        "ap-guangzhou",
        "ap-shanghai",
        "ap-beijing",
        "ap-chengdu",
    ]
    assert logs_schema["properties"]["limit"]["default"] == 20
    assert logs_schema["properties"]["limit"]["minimum"] == 1
    assert logs_schema["properties"]["limit"]["maximum"] == 100
    assert logs.policy.danger_level == DangerLevel.MEDIUM
    assert logs.policy.timeout_seconds == 8
    assert logs.policy.max_retries == 1
    assert logs.policy.idempotent is True


@pytest.mark.asyncio
async def test_business_tool_outputs_are_json_compatible_structures() -> None:
    service = AgentHarnessService.placeholder()
    gateway = service.graph.tool_gateway
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tb",
            "ub",
            "ab",
            "business-tools",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )

    topics = await gateway.execute(
        run_context,
        ModelToolCall(id="topics", name="getAvailableLogTopics", arguments={}),
    )
    alerts = await gateway.execute(
        run_context,
        ModelToolCall(id="alerts", name="queryPrometheusAlerts", arguments={}),
    )
    logs = await gateway.execute(
        run_context,
        ModelToolCall(
            id="logs",
            name="queryLogs",
            arguments={
                "region": "ap-guangzhou",
                "logTopic": "application-logs",
                "query": "level:ERROR",
                "limit": 2,
            },
        ),
    )
    docs = await gateway.execute(
        run_context,
        ModelToolCall(
            id="docs",
            name="queryInternalDocs",
            arguments={"query": "Pod 重启排查流程"},
        ),
    )

    assert topics.result["success"] is True
    assert [topic["topicName"] for topic in topics.result["topics"]] == [
        "system-metrics",
        "application-logs",
        "database-slow-query",
        "system-events",
    ]
    assert topics.result["availableRegions"] == [
        "ap-guangzhou",
        "ap-shanghai",
        "ap-beijing",
        "ap-chengdu",
    ]
    assert topics.result["defaultRegion"] == "ap-guangzhou"

    assert alerts.result["success"] is True
    assert [alert["alertName"] for alert in alerts.result["alerts"]] == [
        "HighCPUUsage",
        "HighMemoryUsage",
        "SlowResponse",
    ]
    assert alerts.result["message"] == "成功检索到 3 个活动告警"

    assert logs.result["success"] is True
    assert logs.result["region"] == "ap-guangzhou"
    assert logs.result["logTopic"] == "application-logs"
    assert logs.result["query"] == "level:ERROR"
    assert logs.result["total"] == 2
    assert logs.result["logs"][0]["level"] == "ERROR"
    assert logs.result["logs"][0]["service"] == "order-service"
    assert isinstance(logs.result["logs"][0]["fields"], dict)

    assert docs.result["status"] == "ok"
    assert docs.result["count"] == 3
    assert docs.result["chunks"][0]["source"] == "runbooks/pod-restart.md"
    assert docs.result["chunks"][0]["ref"] == 1

    events = await service.trace_store.list_by_run("tb", run_context.run_id)
    started = [
        event for event in events if event.event_type == RolloutEventType.TOOL_CALL_STARTED
    ]
    completed = [
        event for event in events if event.event_type == RolloutEventType.TOOL_CALL_COMPLETED
    ]
    assert len(started) == 4
    assert len(completed) == 4
    assert started[0].payload["timeoutSeconds"] == 2
    assert started[0].payload["maxRetries"] == 0
    assert completed[-1].payload["status"] == "success"


@pytest.mark.asyncio
async def test_business_tool_parameter_validation_is_structured() -> None:
    service = AgentHarnessService.placeholder()
    gateway = service.graph.tool_gateway
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tv",
            "uv",
            "av",
            "business-tool-validation",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )

    invalid_region = await gateway.execute(
        run_context,
        ModelToolCall(
            id="invalid-region",
            name="queryLogs",
            arguments={
                "region": "us-west-1",
                "logTopic": "application-logs",
                "query": "level:ERROR",
            },
        ),
    )
    invalid_limit = await gateway.execute(
        run_context,
        ModelToolCall(
            id="invalid-limit",
            name="queryLogs",
            arguments={
                "region": "ap-guangzhou",
                "logTopic": "application-logs",
                "limit": 101,
            },
        ),
    )
    missing_log_topic = await gateway.execute(
        run_context,
        ModelToolCall(
            id="missing-topic",
            name="queryLogs",
            arguments={"region": "ap-guangzhou"},
        ),
    )
    empty_docs_query = await gateway.execute(
        run_context,
        ModelToolCall(
            id="empty-docs-query",
            name="queryInternalDocs",
            arguments={"query": ""},
        ),
    )

    for result in [invalid_region, invalid_limit, missing_log_topic, empty_docs_query]:
        assert result.result["success"] is False
        assert result.result["error_type"] == ToolErrorType.PARAM_VALIDATION_FAILED.value

    events = await service.trace_store.list_by_run("tv", run_context.run_id)
    assert [
        event.event_type for event in events if event.event_type == RolloutEventType.TOOL_CALL_FAILED
    ] == [
        RolloutEventType.TOOL_CALL_FAILED,
        RolloutEventType.TOOL_CALL_FAILED,
        RolloutEventType.TOOL_CALL_FAILED,
        RolloutEventType.TOOL_CALL_FAILED,
    ]


@pytest.mark.asyncio
async def test_internal_docs_no_results_contract() -> None:
    service = AgentHarnessService.placeholder()
    gateway = service.graph.tool_gateway
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "td",
            "ud",
            "ad",
            "docs-no-results",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )

    result = await gateway.execute(
        run_context,
        ModelToolCall(
            id="docs-no-results",
            name="queryInternalDocs",
            arguments={"query": "完全不相关的咖啡菜单"},
        ),
    )

    assert result.result == {
        "status": "no_results",
        "message": "No relevant documents found in the knowledge base.",
    }


@pytest.mark.asyncio
async def test_stub_model_gateway_business_tool_selection_and_final_answers() -> None:
    gateway = StubModelGateway()

    alert_call = await gateway.complete([ModelMessage(role="user", content="现在有哪些活跃告警？")])
    topics_call = await gateway.complete([ModelMessage(role="user", content="有哪些日志主题可以查？")])
    logs_call = await gateway.complete(
        [ModelMessage(role="user", content="查一下 application-logs 里的 ERROR 日志")]
    )
    docs_call = await gateway.complete([ModelMessage(role="user", content="Pod 重启排查流程是什么？")])

    assert alert_call.tool_calls[0].name == "queryPrometheusAlerts"
    assert alert_call.tool_calls[0].arguments == {}
    assert topics_call.tool_calls[0].name == "getAvailableLogTopics"
    assert topics_call.tool_calls[0].arguments == {}
    assert logs_call.tool_calls[0].name == "queryLogs"
    assert logs_call.tool_calls[0].arguments == {
        "region": "ap-guangzhou",
        "logTopic": "application-logs",
        "query": "level:ERROR",
        "limit": 20,
    }
    assert docs_call.tool_calls[0].name == "queryInternalDocs"
    assert docs_call.tool_calls[0].arguments == {"query": "Pod 重启排查流程是什么？"}

    final_alerts = await gateway.complete(
        [
            ModelMessage(role="user", content="现在有哪些活跃告警？"),
            ModelMessage(
                role="tool",
                name="queryPrometheusAlerts",
                content=(
                    '{"success": true, "alerts": ['
                    '{"alertName": "HighCPUUsage"}, {"alertName": "SlowResponse"}]}'
                ),
            ),
        ]
    )
    final_topics = await gateway.complete(
        [
            ModelMessage(role="user", content="有哪些日志主题可以查？"),
            ModelMessage(
                role="tool",
                name="getAvailableLogTopics",
                content=(
                    '{"success": true, "topics": [{"topicName": "application-logs"}], '
                    '"defaultRegion": "ap-guangzhou"}'
                ),
            ),
        ]
    )
    final_logs = await gateway.complete(
        [
            ModelMessage(role="user", content="查一下 application-logs 里的 ERROR 日志"),
            ModelMessage(
                role="tool",
                name="queryLogs",
                content=(
                    '{"success": true, "logTopic": "application-logs", "total": 1, '
                    '"logs": [{"level": "ERROR", "service": "order-service"}]}'
                ),
            ),
        ]
    )
    final_docs = await gateway.complete(
        [
            ModelMessage(role="user", content="Pod 重启排查流程是什么？"),
            ModelMessage(
                role="tool",
                name="queryInternalDocs",
                content=(
                    '{"status": "ok", "count": 1, "chunks": ['
                    '{"ref": 1, "source": "runbooks/pod-restart.md"}]}'
                ),
            ),
        ]
    )

    assert "HighCPUUsage" in final_alerts.content
    assert "application-logs" in final_topics.content
    assert "application-logs" in final_logs.content
    assert "ERROR/order-service" in final_logs.content
    assert "runbooks/pod-restart.md" in final_docs.content


@pytest.mark.asyncio
async def test_stub_model_gateway_ignores_previous_tool_results_for_new_user_message() -> None:
    gateway = StubModelGateway()

    response = await gateway.complete(
        [
            ModelMessage(role="user", content="现在有哪些活跃告警？"),
            ModelMessage(
                role="tool",
                name="queryPrometheusAlerts",
                content='{"success": true, "alerts": [{"alertName": "HighCPUUsage"}]}',
            ),
            ModelMessage(role="assistant", content="当前有 1 个活跃告警：HighCPUUsage。"),
            ModelMessage(role="user", content="hello"),
        ]
    )

    assert response.tool_calls == []
    assert response.content == "[stub] hello"
