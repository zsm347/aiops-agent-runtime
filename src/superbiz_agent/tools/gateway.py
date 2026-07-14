from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from superbiz_agent.config import Settings
from superbiz_agent.harness.context import RunContext
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.stores import RolloutEventStore
from superbiz_agent.model_gateway.base import ModelToolCall
from superbiz_agent.security.redaction import redact_value
from superbiz_agent.security.permissions import PermissionDenied, require_tool_permission
from superbiz_agent.tools.error_translator import (
    MODEL_RETRY_MAX,
    ToolAttemptState,
    ToolErrorTranslator,
    classify_exception,
)
from superbiz_agent.tools.errors import ToolErrorResult, ToolErrorType
from superbiz_agent.tools.policies import ToolAction
from superbiz_agent.tools.registry import ToolDefinition, ToolInvocationContext, ToolRegistry
from superbiz_agent.tools.retry import (
    RetryBackoff,
    RetrySleep,
    default_backoff_seconds,
    default_retry_sleep,
    delay_to_ms,
)


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    result: dict[str, Any]


@dataclass
class _FailureRecord:
    mode: str
    count: int
    controlled_result: ToolErrorResult | None = None


class ToolGateway:
    def __init__(
        self,
        registry: ToolRegistry,
        trace_store: RolloutEventStore,
        *,
        error_translator: ToolErrorTranslator | None = None,
        retry_backoff: RetryBackoff | None = None,
        retry_sleep: RetrySleep | None = None,
        failure_threshold: int = 3,
        settings: Settings | None = None,
    ) -> None:
        self.registry = registry
        self.trace_store = trace_store
        self.error_translator = error_translator or ToolErrorTranslator()
        self.retry_backoff = retry_backoff or default_backoff_seconds
        self.retry_sleep = retry_sleep or default_retry_sleep
        self.failure_threshold = max(1, failure_threshold)
        self.settings = settings
        self._failure_records: dict[tuple[str, str], _FailureRecord] = {}
        self._validation_counts: dict[tuple[str, str, str], int] = {}

    async def block_tool_call(
        self,
        run_context: RunContext,
        tool_call: ModelToolCall,
        *,
        reason: str,
        message: str,
        allowed_next_actions: list[str],
    ) -> ToolExecutionResult:
        error = self.error_translator.translate_blocked(
            tool_name=tool_call.name,
            reason=reason,
            message=message,
            suggestion="Use available evidence and state any unverified gaps.",
        ).model_copy(
            update={
                "allowed_next_actions": self.error_translator.sanitize_allowed_actions(
                    tool_call.name, allowed_next_actions
                ),
                "disallowed_next_actions": [
                    "do_not_claim_the_tool_was_executed",
                    "do_not_invent_missing_evidence",
                ],
            }
        )
        result = self._dump_error(error)
        await self.trace_store.append_event(
            run_context,
            RolloutEventType.TOOL_CALL_BLOCKED,
            {
                "toolName": tool_call.name,
                "reason": reason,
                "result": result,
                "status": "error",
            },
            tool_call_id=tool_call.id,
        )
        return ToolExecutionResult(tool_call.id, tool_call.name, result)

    def clear_run_state(self, run_id: str) -> None:
        self._failure_records = {
            key: value for key, value in self._failure_records.items() if key[0] != run_id
        }
        self._validation_counts = {
            key: value for key, value in self._validation_counts.items() if key[0] != run_id
        }

    async def execute(
        self,
        run_context: RunContext,
        tool_call: ModelToolCall,
    ) -> ToolExecutionResult:
        definition = self.registry.get(tool_call.name)
        if definition is None:
            result = self._dump_error(
                self.error_translator.translate_blocked(
                    tool_name=tool_call.name,
                    reason="unknown_tool",
                    message=f"Unknown tool: {tool_call.name}",
                    suggestion="Use a registered tool.",
                )
            )
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.TOOL_CALL_BLOCKED,
                {
                    "toolName": tool_call.name,
                    "reason": "unknown_tool",
                    "result": result,
                    "status": "error",
                },
                tool_call_id=tool_call.id,
            )
            return ToolExecutionResult(tool_call.id, tool_call.name, result)

        if definition.policy.action == ToolAction.DENY:
            result = self._dump_error(
                self.error_translator.translate_blocked(
                    tool_name=tool_call.name,
                    reason="policy_deny",
                    message=f"Tool is denied by policy: {tool_call.name}",
                    suggestion="Ask for an allowed action.",
                )
            )
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.TOOL_CALL_BLOCKED,
                {
                    "toolName": tool_call.name,
                    "reason": "policy_deny",
                    "result": result,
                    "status": "error",
                },
                tool_call_id=tool_call.id,
            )
            return ToolExecutionResult(tool_call.id, tool_call.name, result)

        permission_result = await self._permission_blocked_result(run_context, tool_call)
        if permission_result is not None:
            return permission_result

        governed_result = self._governed_result_if_execution_blocked(
            run_context,
            tool_call.name,
        )
        if governed_result is not None:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.TOOL_CALL_FAILED,
                {
                    "toolName": tool_call.name,
                    "errorType": governed_result["error_type"],
                    "reason": "run_failure_governance_triggered",
                    "result": governed_result,
                    "status": "error",
                },
                tool_call_id=tool_call.id,
            )
            return ToolExecutionResult(tool_call.id, tool_call.name, governed_result)

        try:
            args = definition.args_model.model_validate(tool_call.arguments)
        except ValidationError as exc:
            model_used = self._increment_validation_count(
                run_context,
                tool_call.name,
                tool_call.id,
            )
            error_result = self.error_translator.translate_validation_error(
                tool_name=tool_call.name,
                args_model=definition.args_model,
                error=exc,
                model_used=model_used,
                model_max=MODEL_RETRY_MAX,
            )
            failure_record = self._record_failure(
                run_context,
                tool_call.name,
                self._validation_failure_mode(exc),
            )
            if failure_record.count >= self.failure_threshold:
                error_result = self.error_translator.translate_governed_failure(
                    tool_name=tool_call.name,
                    failure_mode=failure_record.mode,
                    threshold=self.failure_threshold,
                )
            result = self._dump_error(error_result)
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.TOOL_CALL_FAILED,
                {
                    "toolName": tool_call.name,
                    "errorType": result["error_type"],
                    "result": result,
                    "status": "error",
                },
                tool_call_id=tool_call.id,
            )
            return ToolExecutionResult(tool_call.id, tool_call.name, result)

        await self.trace_store.append_event(
            run_context,
            RolloutEventType.TOOL_CALL_STARTED,
            {
                "toolName": tool_call.name,
                "arguments": redact_value(tool_call.arguments),
                "timeoutSeconds": definition.policy.timeout_seconds,
                "maxRetries": definition.policy.max_retries,
                "idempotent": definition.policy.idempotent,
            },
            tool_call_id=tool_call.id,
        )

        runtime_max = (
            max(0, definition.policy.max_retries) if definition.policy.idempotent else 0
        )
        retries_used = 0
        max_attempts = runtime_max + 1
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self._invoke_handler_with_timeout(
                    definition,
                    args,
                    ToolInvocationContext(
                        run_context=run_context,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                    ),
                )
            except Exception as exc:  # pragma: no cover - uncommon branches covered by 10C tests.
                error_type = classify_exception(exc)
                if error_type == ToolErrorType.TOOL_TIMEOUT:
                    await self.trace_store.append_event(
                        run_context,
                        RolloutEventType.TOOL_CALL_TIMEOUT,
                        {
                            "toolName": tool_call.name,
                            "attempt": attempt,
                            "timeoutSeconds": definition.policy.timeout_seconds,
                            "status": "timeout",
                        },
                        tool_call_id=tool_call.id,
                    )

                if self._should_runtime_retry(error_type, definition, retries_used, runtime_max):
                    retry_number = retries_used + 1
                    backoff_seconds = self.retry_backoff(retry_number)
                    await self.trace_store.append_event(
                        run_context,
                        RolloutEventType.TOOL_CALL_RETRY,
                        {
                            "toolName": tool_call.name,
                            "attempt": attempt,
                            "maxRetries": runtime_max,
                            "errorType": error_type.value,
                            "backoffMs": delay_to_ms(backoff_seconds),
                            "status": "retrying",
                        },
                        tool_call_id=tool_call.id,
                    )
                    retries_used = retry_number
                    await self.retry_sleep(backoff_seconds)
                    continue

                retry_exhausted = (
                    runtime_max > 0
                    and retries_used >= runtime_max
                    and self._is_runtime_retryable_error(error_type)
                    and definition.policy.idempotent
                )
                error_result = self.error_translator.translate_exception(
                    tool_name=tool_call.name,
                    exception=exc,
                    policy=definition.policy,
                    attempt_state=ToolAttemptState(
                        runtime_used=retries_used,
                        runtime_max=runtime_max,
                    ),
                    retry_exhausted=retry_exhausted,
                )
                failure_record = self._record_failure(run_context, tool_call.name, error_type.value)
                if getattr(exc, "retryable_by_model", None) is False:
                    failure_record.controlled_result = error_result
                if (
                    failure_record.count >= self.failure_threshold
                    and getattr(exc, "retryable_by_model", None) is not False
                ):
                    error_result = self.error_translator.translate_governed_failure(
                        tool_name=tool_call.name,
                        failure_mode=failure_record.mode,
                        threshold=self.failure_threshold,
                    )
                result = self._dump_error(error_result)
                await self.trace_store.append_event(
                    run_context,
                    RolloutEventType.TOOL_CALL_FAILED,
                    {
                        "toolName": tool_call.name,
                        "errorType": result["error_type"],
                        "result": result,
                        "attempts": attempt,
                        "status": "error",
                    },
                    tool_call_id=tool_call.id,
                )
                return ToolExecutionResult(tool_call.id, tool_call.name, result)

            self._clear_failure(run_context, tool_call.name)
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.TOOL_CALL_COMPLETED,
                {
                    "toolName": tool_call.name,
                    "result": result,
                    "status": "success",
                },
                tool_call_id=tool_call.id,
            )
            return ToolExecutionResult(tool_call.id, tool_call.name, result)

        raise RuntimeError("Tool execution attempts exhausted without a result.")

    @staticmethod
    def _dump_error(error: ToolErrorResult) -> dict[str, Any]:
        return error.model_dump(mode="json")

    async def _invoke_handler_with_timeout(
        self,
        definition: ToolDefinition,
        args: Any,
        invocation_context: ToolInvocationContext,
    ) -> dict[str, Any]:
        timeout_seconds = max(float(definition.policy.timeout_seconds), 0.001)

        async def invoke_async_handler() -> dict[str, Any]:
            maybe_result = self._call_handler(definition, args, invocation_context)
            return await maybe_result if inspect.isawaitable(maybe_result) else maybe_result

        async def invoke_sync_handler() -> dict[str, Any]:
            maybe_result = await asyncio.to_thread(
                self._call_handler,
                definition,
                args,
                invocation_context,
            )
            return await maybe_result if inspect.isawaitable(maybe_result) else maybe_result

        if inspect.iscoroutinefunction(definition.handler):
            return await asyncio.wait_for(invoke_async_handler(), timeout=timeout_seconds)
        return await asyncio.wait_for(invoke_sync_handler(), timeout=timeout_seconds)

    @staticmethod
    def _call_handler(
        definition: ToolDefinition,
        args: Any,
        invocation_context: ToolInvocationContext,
    ) -> Any:
        if definition.requires_context:
            return definition.handler(args, invocation_context)
        return definition.handler(args)

    def _should_runtime_retry(
        self,
        error_type: ToolErrorType,
        definition: ToolDefinition,
        retries_used: int,
        runtime_max: int,
    ) -> bool:
        return (
            definition.policy.idempotent
            and runtime_max > 0
            and retries_used < runtime_max
            and self._is_runtime_retryable_error(error_type)
        )

    @staticmethod
    def _is_runtime_retryable_error(error_type: ToolErrorType) -> bool:
        return error_type in {
            ToolErrorType.TOOL_TIMEOUT,
            ToolErrorType.UPSTREAM_UNAVAILABLE,
        }

    def _increment_validation_count(
        self,
        run_context: RunContext,
        tool_name: str,
        tool_call_id: str,
    ) -> int:
        key = (run_context.run_id, tool_name, tool_call_id)
        count = self._validation_counts.get(key, 0) + 1
        self._validation_counts[key] = count
        return count

    def _record_failure(
        self,
        run_context: RunContext,
        tool_name: str,
        failure_mode: str,
    ) -> _FailureRecord:
        key = (run_context.run_id, tool_name)
        previous = self._failure_records.get(key)
        if previous and previous.mode == failure_mode:
            previous.count += 1
            return previous

        record = _FailureRecord(mode=failure_mode, count=1)
        self._failure_records[key] = record
        return record

    def _clear_failure(self, run_context: RunContext, tool_name: str) -> None:
        self._failure_records.pop((run_context.run_id, tool_name), None)

    def _governed_result_if_execution_blocked(
        self,
        run_context: RunContext,
        tool_name: str,
    ) -> dict[str, Any] | None:
        record = self._failure_records.get((run_context.run_id, tool_name))
        if record is None or record.count < self.failure_threshold:
            return None
        if record.mode.startswith(f"{ToolErrorType.PARAM_VALIDATION_FAILED.value}:"):
            return None
        if record.controlled_result is not None:
            return self._dump_error(record.controlled_result)
        return self._dump_error(
            self.error_translator.translate_governed_failure(
                tool_name=tool_name,
                failure_mode=record.mode,
                threshold=self.failure_threshold,
            )
        )

    async def _permission_blocked_result(
        self,
        run_context: RunContext,
        tool_call: ModelToolCall,
    ) -> ToolExecutionResult | None:
        try:
            require_tool_permission(
                run_context.request_context,
                tool_call.name,
                settings=self.settings,
            )
        except PermissionDenied as exc:
            result = self._dump_error(
                self.error_translator.translate_blocked(
                    tool_name=tool_call.name,
                    reason="permission_denied",
                    message=str(exc),
                    suggestion="Ask for a permitted action or use an authorized evidence source.",
                )
            )
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.TOOL_CALL_BLOCKED,
                {
                    "toolName": tool_call.name,
                    "reason": "permission_denied",
                    "requiredPermission": exc.required_permission,
                    "result": result,
                    "status": "error",
                },
                tool_call_id=tool_call.id,
            )
            return ToolExecutionResult(tool_call.id, tool_call.name, result)
        return None

    @staticmethod
    def _validation_failure_mode(error: ValidationError) -> str:
        parts = []
        for item in error.errors():
            loc = ".".join(str(part) for part in item.get("loc", ())) or "__root__"
            parts.append(f"{loc}:{item.get('type', 'invalid')}")
        signature = "|".join(sorted(parts)) or "invalid"
        return f"{ToolErrorType.PARAM_VALIDATION_FAILED.value}:{signature}"
