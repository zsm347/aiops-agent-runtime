from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from superbiz_agent.api.app import create_app
from superbiz_agent.config import Settings
from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.model_gateway.base import ModelToolCall
from superbiz_agent.security.auth import AuthContextResolver
from superbiz_agent.security.permissions import (
    CHAT_INVOKE,
    LOCAL_DEFAULT_PERMISSIONS,
    MEMORY_READ,
    MEMORY_WRITE,
    SESSION_CLEAR,
    SESSION_READ,
    TOOL_EXECUTE,
    tool_execute_permission,
)
from superbiz_agent.tools.errors import ToolErrorType


GATEWAY_SECRET = "gateway-secret-for-tests"
DEFAULT_TRUSTED_PERMISSIONS = (
    "chat:invoke,session:read,session:clear,tool:execute,memory:read,memory:write"
)


def _settings(**overrides) -> Settings:
    values = {
        "model_provider": "stub",
        "memory_enabled": False,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def _trusted_headers(
    *,
    tenant_id: str = "tenant-auth",
    user_id: str = "user-auth",
    agent_id: str = "agent-auth",
    permissions: str = DEFAULT_TRUSTED_PERMISSIONS,
    secret: str = GATEWAY_SECRET,
) -> dict[str, str]:
    return {
        "X-Auth-Gateway-Secret": secret,
        "X-Auth-Tenant-Id": tenant_id,
        "X-Auth-User-Id": user_id,
        "X-Auth-Agent-Id": agent_id,
        "X-Auth-Permissions": permissions,
        "X-Auth-Roles": "operator",
    }


def test_local_dev_headers_are_accepted_and_get_default_permissions() -> None:
    app = create_app(_settings(app_env="local", auth_mode="dev_headers"))
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"Id": "local-dev-session", "Question": "hello"},
        headers={
            "X-Tenant-Id": "tenant-local",
            "X-User-Id": "user-local",
            "X-Agent-Id": "agent-local",
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["success"] is True

    context = AgentRequestContext("tenant-local", "user-local", "agent-local", "local-dev-session")
    events = asyncio.run(app.state.harness_service.trace_store.list_by_session(context))
    run_started = next(
        event for event in events if event.event_type == RolloutEventType.RUN_STARTED
    )
    assert run_started.payload["authMode"] == "dev_headers"

    principal = AuthContextResolver(_settings(app_env="local")).resolve({})
    assert principal.permissions == LOCAL_DEFAULT_PERMISSIONS


def test_production_rejects_dev_headers_mode() -> None:
    app = create_app(_settings(app_env="production", auth_mode="dev_headers"))
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"Id": "prod-dev", "Question": "hello"},
        headers={
            "X-Tenant-Id": "tenant-forged",
            "X-User-Id": "user-forged",
            "X-Agent-Id": "agent-forged",
        },
    )

    assert response.status_code == 401
    assert "dev_headers" in response.json()["detail"]


def test_production_rejects_permission_enforcement_bypass() -> None:
    app = create_app(
        _settings(
            app_env="production",
            auth_mode="trusted_gateway",
            auth_trusted_gateway_secret=GATEWAY_SECRET,
            auth_permission_enforcement=False,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"Id": "prod-bypass", "Question": "hello"},
        headers=_trusted_headers(),
    )

    assert response.status_code == 401
    assert "auth_permission_enforcement=false" in response.json()["detail"]


def test_jwt_mode_returns_explicit_p0_auth_error() -> None:
    app = create_app(_settings(app_env="production", auth_mode="jwt"))
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"Id": "jwt-p0", "Question": "hello"},
        headers={"Authorization": "Bearer test.jwt.token"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "JWT auth not implemented in P0."


def test_trusted_gateway_requires_configured_and_present_secret() -> None:
    missing_config_app = create_app(_settings(app_env="production", auth_mode="trusted_gateway"))
    missing_config = TestClient(missing_config_app).post(
        "/api/chat",
        json={"Id": "missing-config", "Question": "hello"},
        headers=_trusted_headers(),
    )

    configured_app = create_app(
        _settings(
            app_env="production",
            auth_mode="trusted_gateway",
            auth_trusted_gateway_secret=GATEWAY_SECRET,
        )
    )
    missing_header = TestClient(configured_app).post(
        "/api/chat",
        json={"Id": "missing-header", "Question": "hello"},
        headers={
            key: value
            for key, value in _trusted_headers().items()
            if key != "X-Auth-Gateway-Secret"
        },
    )

    assert missing_config.status_code == 401
    assert "auth_trusted_gateway_secret" in missing_config.json()["detail"]
    assert missing_header.status_code == 401
    assert "Invalid trusted gateway secret" in missing_header.json()["detail"]


def test_trusted_gateway_identity_cannot_be_overridden_by_regular_headers() -> None:
    app = create_app(
        _settings(
            app_env="production",
            auth_mode="trusted_gateway",
            auth_trusted_gateway_secret=GATEWAY_SECRET,
        )
    )
    client = TestClient(app)
    headers = {
        **_trusted_headers(
            tenant_id="tenant-trusted",
            user_id="user-trusted",
            agent_id="agent-trusted",
            permissions=CHAT_INVOKE,
        ),
        "X-Tenant-Id": "tenant-forged",
        "X-User-Id": "user-forged",
        "X-Agent-Id": "agent-forged",
    }

    response = client.post(
        "/api/chat",
        json={"Id": "trusted-session", "Question": "hello"},
        headers=headers,
    )

    assert response.status_code == 200
    trusted_context = AgentRequestContext(
        "tenant-trusted",
        "user-trusted",
        "agent-trusted",
        "trusted-session",
    )
    forged_context = AgentRequestContext(
        "tenant-forged",
        "user-forged",
        "agent-forged",
        "trusted-session",
    )
    trusted_events = asyncio.run(
        app.state.harness_service.trace_store.list_by_session(trusted_context)
    )
    forged_events = asyncio.run(
        app.state.harness_service.trace_store.list_by_session(forged_context)
    )

    assert trusted_events
    assert forged_events == []
    assert trusted_events[0].payload["authMode"] == "trusted_gateway"


def test_api_permission_denied_returns_403_for_chat_and_clear() -> None:
    app = create_app(
        _settings(
            app_env="production",
            auth_mode="trusted_gateway",
            auth_trusted_gateway_secret=GATEWAY_SECRET,
        )
    )
    client = TestClient(app)

    chat_denied = client.post(
        "/api/chat",
        json={"Id": "no-chat", "Question": "hello"},
        headers=_trusted_headers(permissions=SESSION_READ),
    )
    clear_denied = client.post(
        "/api/chat/clear",
        json={"Id": "no-clear"},
        headers=_trusted_headers(permissions=CHAT_INVOKE),
    )

    assert chat_denied.status_code == 403
    assert "chat:invoke" in chat_denied.json()["detail"]
    assert clear_denied.status_code == 403
    assert "session:clear" in clear_denied.json()["detail"]


@pytest.mark.asyncio
async def test_tool_gateway_blocks_missing_tool_execute_permission() -> None:
    service = AgentHarnessService.build_default(_settings(app_env="local", memory_enabled=True))
    gateway = service.graph.tool_gateway

    denied_run = await service.runtime.start_run(
        AgentRequestContext("tenant-tool", "user-tool", "agent-tool", "tool-denied")
    )
    denied = await gateway.execute(
        denied_run,
        ModelToolCall(id="tool-denied", name="getCurrentDateTime", arguments={}),
    )

    allowed_run = await service.runtime.start_run(
        AgentRequestContext(
            "tenant-tool",
            "user-tool",
            "agent-tool",
            "tool-allowed",
            permissions=frozenset({tool_execute_permission("getCurrentDateTime")}),
            auth_mode="dev_headers",
        )
    )
    allowed = await gateway.execute(
        allowed_run,
        ModelToolCall(id="tool-allowed", name="getCurrentDateTime", arguments={}),
    )

    assert denied.result["success"] is False
    assert denied.result["error_type"] == ToolErrorType.TOOL_BLOCKED.value
    assert denied.result["reason"] == "permission_denied"
    denied_events = await service.trace_store.list_by_run("tenant-tool", denied_run.run_id)
    assert denied_events[-1].event_type == RolloutEventType.TOOL_CALL_BLOCKED

    assert allowed.result["success"] is True
    assert allowed.result["data"]["timezone"] == "Asia/Shanghai"


@pytest.mark.asyncio
async def test_memory_write_tool_is_blocked_without_memory_write_permission() -> None:
    service = AgentHarnessService.build_default(_settings(app_env="local", memory_enabled=True))
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tenant-memory",
            "user-memory",
            "agent-memory",
            "memory-denied",
            permissions=frozenset({MEMORY_READ, TOOL_EXECUTE}),
            auth_mode="dev_headers",
        )
    )

    result = await service.graph.tool_gateway.execute(
        run_context,
        ModelToolCall(
            id="memory-write",
            name="updateCoreMemory",
            arguments={
                "blockKey": "user_rules",
                "newContent": "以后按证据优先回答。",
            },
        ),
    )

    assert result.result["success"] is False
    assert result.result["error_type"] == ToolErrorType.TOOL_BLOCKED.value
    assert result.result["reason"] == "permission_denied"


def test_tenant_b_cannot_read_or_clear_tenant_a_session() -> None:
    app = create_app(
        _settings(
            app_env="production",
            auth_mode="trusted_gateway",
            auth_trusted_gateway_secret=GATEWAY_SECRET,
        )
    )
    client = TestClient(app)
    permissions = f"{CHAT_INVOKE},{SESSION_READ},{SESSION_CLEAR}"

    client.post(
        "/api/chat",
        json={"Id": "shared-session", "Question": "hello"},
        headers=_trusted_headers(tenant_id="tenant-a", permissions=permissions),
    )
    tenant_b_read = client.get(
        "/api/chat/session/shared-session",
        headers=_trusted_headers(tenant_id="tenant-b", permissions=permissions),
    )
    tenant_b_clear = client.post(
        "/api/chat/clear",
        json={"Id": "shared-session"},
        headers=_trusted_headers(tenant_id="tenant-b", permissions=permissions),
    )
    tenant_a_read = client.get(
        "/api/chat/session/shared-session",
        headers=_trusted_headers(tenant_id="tenant-a", permissions=permissions),
    )

    assert tenant_b_read.status_code == 200
    assert tenant_b_read.json()["data"]["messagePairCount"] == 0
    assert tenant_b_clear.status_code == 200
    assert tenant_a_read.json()["data"]["messagePairCount"] == 1


def test_gateway_secret_and_regular_auth_headers_do_not_enter_trace() -> None:
    secret = "super-secret-gateway-token"
    app = create_app(
        _settings(
            app_env="production",
            auth_mode="trusted_gateway",
            auth_trusted_gateway_secret=secret,
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"Id": "trace-secret", "Question": "hello"},
        headers=_trusted_headers(secret=secret, permissions=CHAT_INVOKE),
    )

    assert response.status_code == 200
    context = AgentRequestContext("tenant-auth", "user-auth", "agent-auth", "trace-secret")
    events = asyncio.run(app.state.harness_service.trace_store.list_by_session(context))
    trace_text = json.dumps([event.model_dump(mode="json") for event in events], ensure_ascii=False)

    assert secret not in trace_text
    assert "X-Auth-Gateway-Secret" not in trace_text
    assert "trusted_gateway" in trace_text


@pytest.mark.asyncio
async def test_tool_arguments_are_redacted_before_trace() -> None:
    service = AgentHarnessService.build_default(_settings(app_env="local", memory_enabled=True))
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tenant-redact",
            "user-redact",
            "agent-redact",
            "tool-arg-redact",
            permissions=frozenset({MEMORY_WRITE}),
            auth_mode="dev_headers",
        )
    )

    await service.graph.tool_gateway.execute(
        run_context,
        ModelToolCall(
            id="secret-arg",
            name="saveArchivalMemory",
            arguments={
                "topic": "secret",
                "content": "password=super-secret-value",
            },
        ),
    )

    events = await service.trace_store.list_by_run("tenant-redact", run_context.run_id)
    trace_text = json.dumps([event.model_dump(mode="json") for event in events], ensure_ascii=False)

    assert "super-secret-value" not in trace_text
    assert "password=[REDACTED]" in trace_text


def test_no_raw_ref_or_runtime_identity_parameters_are_exposed_in_tool_schemas() -> None:
    service = AgentHarnessService.build_default(_settings(app_env="local", memory_enabled=True))
    forbidden = {
        "tenantId",
        "tenant_id",
        "userId",
        "user_id",
        "runId",
        "run_id",
        "rawRef",
        "raw_ref",
    }

    for tool in service.graph.tool_gateway.registry.list():
        schema = tool.args_model.model_json_schema()
        assert not forbidden.intersection(schema.get("properties", {}))
        assert "raw" not in tool.name.lower()
