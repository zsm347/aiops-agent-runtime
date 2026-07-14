from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.model_gateway.base import ModelToolCall
from superbiz_agent.security.permissions import LOCAL_DEFAULT_PERMISSIONS
from superbiz_agent.tools.errors import ToolErrorType
from superbiz_agent.tools.gateway import ToolGateway
from superbiz_agent.tools.policies import DangerLevel, ToolPolicy
from superbiz_agent.tools.registry import ToolDefinition, ToolRegistry


class ReliabilityArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    region: Literal["ap-guangzhou", "ap-shanghai"]
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


async def _no_sleep(_: float) -> None:
    return None


def _registry(definition: ToolDefinition) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(definition)
    return registry


def _definition(
    handler,
    *,
    name: str = "reliableTool",
    timeout_seconds: int = 1,
    max_retries: int = 0,
    idempotent: bool = True,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Reliability test tool.",
        args_model=ReliabilityArgs,
        handler=handler,
        policy=ToolPolicy(
            tool_name=name,
            danger_level=DangerLevel.LOW,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            idempotent=idempotent,
        ),
    )


async def _run_context(session_id: str):
    service = AgentHarnessService.placeholder()
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tenant-r",
            "user-r",
            "agent-r",
            session_id,
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    return service, run_context


def _valid_arguments() -> dict:
    return {"region": "ap-guangzhou", "query": "errors", "limit": 5}


@pytest.mark.asyncio
async def test_validation_failure_returns_field_violations_without_handler_or_retry() -> None:
    calls = 0

    async def handler(_: ReliabilityArgs) -> dict:
        nonlocal calls
        calls += 1
        return {"success": True}

    service, run_context = await _run_context("reliability-validation")
    gateway = ToolGateway(_registry(_definition(handler, max_retries=2)), service.trace_store)

    result = await gateway.execute(
        run_context,
        ModelToolCall(
            id="invalid",
            name="reliableTool",
            arguments={"region": "us-west-1", "query": "", "limit": 101},
        ),
    )

    assert calls == 0
    assert result.result["success"] is False
    assert result.result["error_type"] == ToolErrorType.PARAM_VALIDATION_FAILED.value
    assert result.result["retryable_by_runtime"] is False
    assert result.result["retryable_by_model"] is True
    assert "retry_same_tool_with_fixed_arguments" in result.result["allowed_next_actions"]
    violations = {violation["field"]: violation for violation in result.result["violations"]}
    assert violations["region"]["received"] == "us-west-1"
    assert "ap-guangzhou" in violations["region"]["expected"]
    assert violations["query"]["fix_hint"]
    assert violations["limit"]["received"] == 101
    assert result.result["retry_budget"] == {
        "runtime_used": 0,
        "runtime_max": 0,
        "model_used": 1,
        "model_max": 2,
    }

    events = await service.trace_store.list_by_run("tenant-r", run_context.run_id)
    assert RolloutEventType.TOOL_CALL_STARTED not in [event.event_type for event in events]
    assert [event.event_type for event in events[-1:]] == [RolloutEventType.TOOL_CALL_FAILED]


@pytest.mark.asyncio
async def test_validation_model_retry_budget_is_exhausted_for_same_tool_call() -> None:
    async def handler(_: ReliabilityArgs) -> dict:
        return {"success": True}

    service, run_context = await _run_context("reliability-validation-budget")
    gateway = ToolGateway(_registry(_definition(handler)), service.trace_store)
    invalid_call = ModelToolCall(
        id="same-invalid-call",
        name="reliableTool",
        arguments={"region": "us-west-1", "query": "errors", "limit": 5},
    )

    first = await gateway.execute(run_context, invalid_call)
    second = await gateway.execute(run_context, invalid_call)

    assert first.result["retryable_by_model"] is True
    assert first.result["retry_budget"]["model_used"] == 1
    assert second.result["retryable_by_model"] is False
    assert second.result["retry_budget"]["model_used"] == 2
    assert second.result["allowed_next_actions"] == [
        "ask_user_for_missing_required_fields",
        "degrade_with_user_friendly_error",
    ]


@pytest.mark.asyncio
async def test_timeout_seconds_is_enforced_for_async_handler_and_traced() -> None:
    async def slow_handler(_: ReliabilityArgs) -> dict:
        await asyncio.sleep(0.05)
        return {"success": True}

    service, run_context = await _run_context("reliability-timeout")
    gateway = ToolGateway(
        _registry(_definition(slow_handler, timeout_seconds=0, max_retries=0)),
        service.trace_store,
    )

    result = await gateway.execute(
        run_context,
        ModelToolCall(id="timeout", name="reliableTool", arguments=_valid_arguments()),
    )

    assert result.result["success"] is False
    assert result.result["error_type"] == ToolErrorType.TOOL_TIMEOUT.value
    assert result.result["retryable_by_runtime"] is False
    events = await service.trace_store.list_by_run("tenant-r", run_context.run_id)
    event_types = [event.event_type for event in events]
    assert RolloutEventType.TOOL_CALL_TIMEOUT in event_types
    assert RolloutEventType.TOOL_CALL_RETRY not in event_types
    assert events[-1].event_type == RolloutEventType.TOOL_CALL_FAILED


@pytest.mark.asyncio
async def test_idempotent_tool_runtime_retries_and_eventually_succeeds() -> None:
    attempts = 0
    sleeps: list[float] = []

    async def flaky_handler(_: ReliabilityArgs) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("upstream 503 temporarily unavailable")
        return {"success": True, "attempts": attempts}

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    service, run_context = await _run_context("reliability-retry-success")
    gateway = ToolGateway(
        _registry(_definition(flaky_handler, max_retries=1, idempotent=True)),
        service.trace_store,
        retry_backoff=lambda retry_number: retry_number * 0.01,
        retry_sleep=record_sleep,
    )

    result = await gateway.execute(
        run_context,
        ModelToolCall(id="retry-success", name="reliableTool", arguments=_valid_arguments()),
    )

    assert result.result == {"success": True, "attempts": 2}
    assert sleeps == [0.01]
    events = await service.trace_store.list_by_run("tenant-r", run_context.run_id)
    retry_events = [event for event in events if event.event_type == RolloutEventType.TOOL_CALL_RETRY]
    assert len(retry_events) == 1
    assert retry_events[0].payload["errorType"] == ToolErrorType.UPSTREAM_UNAVAILABLE.value
    assert retry_events[0].payload["backoffMs"] == 10
    assert events[-1].event_type == RolloutEventType.TOOL_CALL_COMPLETED
    started = [event for event in events if event.event_type == RolloutEventType.TOOL_CALL_STARTED]
    assert started[-1].payload["idempotent"] is True


@pytest.mark.asyncio
async def test_retry_exhaustion_returns_structured_error() -> None:
    attempts = 0

    async def always_fails(_: ReliabilityArgs) -> dict:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("connection reset by peer")

    service, run_context = await _run_context("reliability-retry-exhausted")
    gateway = ToolGateway(
        _registry(_definition(always_fails, max_retries=2, idempotent=True)),
        service.trace_store,
        retry_backoff=lambda _: 0,
        retry_sleep=_no_sleep,
    )

    result = await gateway.execute(
        run_context,
        ModelToolCall(id="retry-exhausted", name="reliableTool", arguments=_valid_arguments()),
    )

    assert attempts == 3
    assert result.result["success"] is False
    assert result.result["error_type"] == ToolErrorType.TOOL_RETRY_EXHAUSTED.value
    assert result.result["reason"] == "runtime_retry_budget_exhausted"
    assert result.result["retry_budget"]["runtime_used"] == 2
    assert result.result["retry_budget"]["runtime_max"] == 2
    events = await service.trace_store.list_by_run("tenant-r", run_context.run_id)
    assert [event.event_type for event in events].count(RolloutEventType.TOOL_CALL_RETRY) == 2
    assert events[-1].event_type == RolloutEventType.TOOL_CALL_FAILED
    assert events[-1].payload["attempts"] == 3


@pytest.mark.asyncio
async def test_non_idempotent_tool_does_not_runtime_retry() -> None:
    attempts = 0

    async def failing_mutation(_: ReliabilityArgs) -> dict:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("upstream 503 temporarily unavailable")

    service, run_context = await _run_context("reliability-non-idempotent")
    gateway = ToolGateway(
        _registry(_definition(failing_mutation, max_retries=3, idempotent=False)),
        service.trace_store,
        retry_backoff=lambda _: 0,
        retry_sleep=_no_sleep,
    )

    result = await gateway.execute(
        run_context,
        ModelToolCall(id="non-idempotent", name="reliableTool", arguments=_valid_arguments()),
    )

    assert attempts == 1
    assert result.result["success"] is False
    assert result.result["error_type"] == ToolErrorType.UPSTREAM_UNAVAILABLE.value
    assert result.result["retryable_by_runtime"] is False
    assert result.result["retry_budget"]["runtime_max"] == 0
    events = await service.trace_store.list_by_run("tenant-r", run_context.run_id)
    assert RolloutEventType.TOOL_CALL_RETRY not in [event.event_type for event in events]


@pytest.mark.asyncio
async def test_run_local_failure_governance_stops_blind_handler_execution() -> None:
    attempts = 0

    async def always_times_out(_: ReliabilityArgs) -> dict:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("timed out")

    service, run_context = await _run_context("reliability-governance-a")
    gateway = ToolGateway(
        _registry(_definition(always_times_out, max_retries=0, idempotent=True)),
        service.trace_store,
        retry_sleep=_no_sleep,
        failure_threshold=3,
    )

    results = []
    for index in range(4):
        results.append(
            await gateway.execute(
                run_context,
                ModelToolCall(
                    id=f"governed-{index}",
                    name="reliableTool",
                    arguments=_valid_arguments(),
                ),
            )
        )

    assert attempts == 3
    assert results[0].result["error_type"] == ToolErrorType.TOOL_TIMEOUT.value
    assert results[2].result["error_type"] == ToolErrorType.TOOL_RETRY_EXHAUSTED.value
    assert results[2].result["reason"] == "run_failure_governance_triggered"
    assert results[3].result["reason"] == "run_failure_governance_triggered"

    other_run = await service.runtime.start_run(
        AgentRequestContext(
            "tenant-r",
            "user-r",
            "agent-r",
            "reliability-governance-b",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    await gateway.execute(
        other_run,
        ModelToolCall(id="other-run", name="reliableTool", arguments=_valid_arguments()),
    )
    assert attempts == 4


@pytest.mark.asyncio
async def test_exception_messages_and_trace_results_are_sanitized() -> None:
    async def leaks_secret(_: ReliabilityArgs) -> dict:
        raise RuntimeError(
            "Traceback (most recent call last): File "
            "/Users/zsm/private/service.py password=hunter2 "
            "api_key=sk-live-secret token=abc123"
        )

    service, run_context = await _run_context("reliability-redaction")
    gateway = ToolGateway(
        _registry(_definition(leaks_secret, max_retries=0, idempotent=True)),
        service.trace_store,
    )

    result = await gateway.execute(
        run_context,
        ModelToolCall(id="redacted", name="reliableTool", arguments=_valid_arguments()),
    )

    result_text = json.dumps(result.result, ensure_ascii=False)
    assert "hunter2" not in result_text
    assert "sk-live-secret" not in result_text
    assert "abc123" not in result_text
    assert "/Users/zsm/private/service.py" not in result_text
    assert "redacted" in result_text

    events = await service.trace_store.list_by_run("tenant-r", run_context.run_id)
    trace_text = json.dumps(events[-1].payload["result"], ensure_ascii=False)
    assert "hunter2" not in trace_text
    assert "sk-live-secret" not in trace_text
    assert "abc123" not in trace_text
    assert "/Users/zsm/private/service.py" not in trace_text


@pytest.mark.asyncio
async def test_timeout_seconds_is_enforced_for_sync_handler() -> None:
    def slow_sync_handler(_: ReliabilityArgs) -> dict:
        time.sleep(0.05)
        return {"success": True}

    service, run_context = await _run_context("reliability-sync-timeout")
    gateway = ToolGateway(
        _registry(_definition(slow_sync_handler, timeout_seconds=0, max_retries=0)),
        service.trace_store,
    )

    result = await gateway.execute(
        run_context,
        ModelToolCall(id="sync-timeout", name="reliableTool", arguments=_valid_arguments()),
    )

    assert result.result["success"] is False
    assert result.result["error_type"] == ToolErrorType.TOOL_TIMEOUT.value
    events = await service.trace_store.list_by_run("tenant-r", run_context.run_id)
    assert RolloutEventType.TOOL_CALL_TIMEOUT in [event.event_type for event in events]
