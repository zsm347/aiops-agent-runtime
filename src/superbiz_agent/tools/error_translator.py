from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from superbiz_agent.security.redaction import sanitize_error_message
from superbiz_agent.tools.errors import (
    ToolErrorResult,
    ToolErrorType,
    ToolRetryBudget,
    ToolViolation,
)
from superbiz_agent.tools.policies import ToolPolicy


MODEL_RETRY_MAX = 2


@dataclass(frozen=True)
class ToolAttemptState:
    runtime_used: int = 0
    runtime_max: int = 0
    model_used: int = 0
    model_max: int = MODEL_RETRY_MAX

    def budget(self) -> ToolRetryBudget:
        return ToolRetryBudget(
            runtime_used=self.runtime_used,
            runtime_max=self.runtime_max,
            model_used=self.model_used,
            model_max=self.model_max,
        )


class ToolErrorTranslator:
    def translate_validation_error(
        self,
        *,
        tool_name: str,
        args_model: type[BaseModel],
        error: ValidationError,
        model_used: int,
        model_max: int = MODEL_RETRY_MAX,
    ) -> ToolErrorResult:
        violations = [
            self._violation_from_pydantic_error(args_model, item) for item in error.errors()
        ]
        retryable_by_model = model_used < model_max
        allowed_next_actions = (
            [
                "retry_same_tool_with_fixed_arguments",
                "ask_user_for_missing_required_fields",
            ]
            if retryable_by_model
            else [
                "ask_user_for_missing_required_fields",
                "degrade_with_user_friendly_error",
            ]
        )
        return ToolErrorResult(
            error_type=ToolErrorType.PARAM_VALIDATION_FAILED,
            reason="tool_arguments_invalid",
            message=(
                f"{tool_name} tool arguments are invalid. Use violations to fix fields; "
                "ask the user if required information is missing."
            ),
            suggestion="Fix the tool arguments according to violations, or ask the user.",
            violations=violations,
            retryable_by_runtime=False,
            retryable_by_model=retryable_by_model,
            allowed_next_actions=self.sanitize_allowed_actions(tool_name, allowed_next_actions),
            disallowed_next_actions=[
                "do_not_guess_missing_required_fields",
                "do_not_call_unrelated_tools",
            ],
            retry_budget=ToolRetryBudget(
                runtime_used=0,
                runtime_max=0,
                model_used=model_used,
                model_max=model_max,
            ),
        )

    def translate_blocked(
        self,
        *,
        tool_name: str,
        reason: str,
        message: str,
        suggestion: str,
    ) -> ToolErrorResult:
        return ToolErrorResult(
            error_type=ToolErrorType.TOOL_BLOCKED,
            reason=reason,
            message=sanitize_error_message(message),
            suggestion=suggestion,
            retryable_by_runtime=False,
            retryable_by_model=False,
            allowed_next_actions=self.sanitize_allowed_actions(tool_name, [
                "use_alternative_tool",
                "search_memory_for_historical_reference",
                "search_rag_for_runbook",
                "degrade_with_user_friendly_error",
            ]),
            disallowed_next_actions=[
                "do_not_retry_blocked_tool",
                "do_not_claim_tool_result_without_evidence",
            ],
            retry_budget=ToolRetryBudget(runtime_used=0, runtime_max=0, model_used=0, model_max=0),
        )

    def translate_governed_failure(
        self,
        *,
        tool_name: str,
        failure_mode: str,
        threshold: int,
    ) -> ToolErrorResult:
        return ToolErrorResult(
            error_type=ToolErrorType.TOOL_RETRY_EXHAUSTED,
            reason="run_failure_governance_triggered",
            message=(
                f"{tool_name} has repeatedly failed with {failure_mode} in this run. "
                f"After {threshold} consecutive failures, blind retries are stopped."
            ),
            suggestion="Use another evidence source, ask the user, or provide a cautious degraded answer.",
            retryable_by_runtime=False,
            retryable_by_model=False,
            allowed_next_actions=self.sanitize_allowed_actions(tool_name, [
                "use_alternative_tool",
                "search_memory_for_historical_reference",
                "search_rag_for_runbook",
                "ask_user_for_missing_required_fields",
                "degrade_with_unconfirmed_items",
            ]),
            disallowed_next_actions=[
                "do_not_retry_after_budget_exhausted",
                "do_not_claim_tool_result_without_evidence",
            ],
            retry_budget=ToolRetryBudget(runtime_used=0, runtime_max=0, model_used=0, model_max=0),
        )

    def translate_exception(
        self,
        *,
        tool_name: str,
        exception: BaseException,
        policy: ToolPolicy,
        attempt_state: ToolAttemptState,
        retry_exhausted: bool = False,
    ) -> ToolErrorResult:
        base_error_type = classify_exception(exception)
        error_type = ToolErrorType.TOOL_RETRY_EXHAUSTED if retry_exhausted else base_error_type
        transient = base_error_type in {
            ToolErrorType.TOOL_TIMEOUT,
            ToolErrorType.UPSTREAM_UNAVAILABLE,
        }
        retryable_by_runtime = (
            transient
            and policy.idempotent
            and attempt_state.runtime_used < attempt_state.runtime_max
            and not retry_exhausted
        )
        retryable_by_model = base_error_type not in {ToolErrorType.PERMISSION_DENIED}
        declared_runtime_retry = getattr(exception, "retryable", None)
        if declared_runtime_retry is False:
            retryable_by_runtime = False
        declared_model_retry = getattr(exception, "retryable_by_model", None)
        if declared_model_retry is False:
            retryable_by_model = False
        reason = _reason_for_error(error_type, base_error_type, retry_exhausted)
        message = _message_for_exception(tool_name, exception, base_error_type, retry_exhausted)
        allowed_next_actions = _controlled_allowed_actions(exception) or _allowed_actions_for_error(
            base_error_type, retry_exhausted
        )
        return ToolErrorResult(
            error_type=error_type,
            reason=reason,
            message=message,
            suggestion=_suggestion_for_error(base_error_type, retry_exhausted),
            retryable_by_runtime=retryable_by_runtime,
            retryable_by_model=retryable_by_model and not retry_exhausted,
            allowed_next_actions=self.sanitize_allowed_actions(
                tool_name, allowed_next_actions
            ),
            disallowed_next_actions=_disallowed_actions_for_error(retry_exhausted),
            retry_budget=attempt_state.budget(),
        )

    @staticmethod
    def sanitize_allowed_actions(tool_name: str, actions: list[str]) -> list[str]:
        if tool_name == "queryInternalDocs":
            return [action for action in actions if action != "search_rag_for_runbook"]
        return list(actions)

    def _violation_from_pydantic_error(
        self,
        args_model: type[BaseModel],
        error: dict[str, Any],
    ) -> ToolViolation:
        field = _field_path(error.get("loc", ()))
        problem = sanitize_error_message(str(error.get("msg") or "Invalid value."))
        received = _json_safe_value(error.get("input"))
        expected = _expected_from_schema(args_model.model_json_schema(), field)
        fix_hint = _fix_hint(field, str(error.get("type") or ""), expected)
        return ToolViolation(
            field=field,
            problem=problem,
            expected=expected,
            received=received,
            fix_hint=fix_hint,
        )


def classify_exception(exception: BaseException) -> ToolErrorType:
    if isinstance(exception, (asyncio.TimeoutError, TimeoutError)):
        return ToolErrorType.TOOL_TIMEOUT
    if isinstance(exception, PermissionError):
        return ToolErrorType.PERMISSION_DENIED

    text = str(exception).lower()
    status = getattr(exception, "status_code", None) or getattr(exception, "status", None)
    if status in {401, 403} or "permission denied" in text or "forbidden" in text:
        return ToolErrorType.PERMISSION_DENIED
    if status in {429, 502, 503, 504}:
        return ToolErrorType.UPSTREAM_UNAVAILABLE
    if any(marker in text for marker in ("429", "502", "503", "504", "rate limit")):
        return ToolErrorType.UPSTREAM_UNAVAILABLE
    if any(
        marker in text
        for marker in (
            "connection refused",
            "connection reset",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "upstream unavailable",
            "service unavailable",
        )
    ):
        return ToolErrorType.UPSTREAM_UNAVAILABLE
    return ToolErrorType.TOOL_ERROR


def _field_path(loc: Any) -> str:
    if not loc:
        return "__root__"
    if isinstance(loc, (list, tuple)):
        return ".".join(str(part) for part in loc)
    return str(loc)


def _expected_from_schema(schema: dict[str, Any], field: str) -> str | None:
    current_schema = schema
    for part in field.split("."):
        properties = current_schema.get("properties")
        if not isinstance(properties, dict):
            return None
        current_schema = properties.get(part)
        if not isinstance(current_schema, dict):
            return None

    if "enum" in current_schema:
        return "one of: " + ", ".join(str(item) for item in current_schema["enum"])

    constraints: list[str] = []
    if "type" in current_schema:
        constraints.append(str(current_schema["type"]))
    if "minimum" in current_schema and "maximum" in current_schema:
        constraints.append(f"{current_schema['minimum']} to {current_schema['maximum']}")
    elif "minimum" in current_schema:
        constraints.append(f">= {current_schema['minimum']}")
    elif "maximum" in current_schema:
        constraints.append(f"<= {current_schema['maximum']}")
    if "minLength" in current_schema:
        constraints.append(f"min length {current_schema['minLength']}")
    if "maxLength" in current_schema:
        constraints.append(f"max length {current_schema['maxLength']}")
    if "default" in current_schema:
        constraints.append(f"default {current_schema['default']}")
    return ", ".join(constraints) if constraints else None


def _fix_hint(field: str, error_type: str, expected: str | None) -> str:
    if error_type == "missing":
        return f"Provide {field}; ask the user if it is not available."
    if "enum" in error_type or (expected and expected.startswith("one of:")):
        return f"Choose {field} from the allowed values."
    if "less_than_equal" in error_type or "greater_than_equal" in error_type:
        return f"Adjust {field} to fit the allowed range."
    if "too_short" in error_type or "string_too_short" in error_type:
        return f"Provide a non-empty value for {field}."
    if expected:
        return f"Set {field} to match: {expected}."
    return f"Correct {field} according to the tool schema."


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return sanitize_error_message(value) if isinstance(value, str) else value
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in list(value.items())[:20]}
    return sanitize_error_message(repr(value))


def _reason_for_error(
    error_type: ToolErrorType,
    base_error_type: ToolErrorType,
    retry_exhausted: bool,
) -> str:
    if retry_exhausted:
        return "runtime_retry_budget_exhausted"
    if error_type == ToolErrorType.TOOL_TIMEOUT:
        return "tool_execution_timeout"
    if base_error_type == ToolErrorType.UPSTREAM_UNAVAILABLE:
        return "upstream_temporarily_unavailable"
    if base_error_type == ToolErrorType.PERMISSION_DENIED:
        return "tool_permission_denied"
    return "tool_execution_failed"


def _message_for_exception(
    tool_name: str,
    exception: BaseException,
    base_error_type: ToolErrorType,
    retry_exhausted: bool,
) -> str:
    detail = sanitize_error_message(str(exception))
    if retry_exhausted:
        return f"{tool_name} failed after runtime retries were exhausted. Last safe detail: {detail}"
    if base_error_type == ToolErrorType.TOOL_TIMEOUT:
        return f"{tool_name} timed out before the configured deadline."
    if base_error_type == ToolErrorType.UPSTREAM_UNAVAILABLE:
        return f"{tool_name} could not reach an upstream dependency. Safe detail: {detail}"
    if base_error_type == ToolErrorType.PERMISSION_DENIED:
        return f"{tool_name} cannot be used because permission was denied."
    return f"{tool_name} failed during execution. Safe detail: {detail}"


def _suggestion_for_error(base_error_type: ToolErrorType, retry_exhausted: bool) -> str:
    if retry_exhausted:
        return "Stop automatic retries and use another evidence path or provide a cautious answer."
    if base_error_type == ToolErrorType.TOOL_TIMEOUT:
        return "Retry with a narrower scope, use another tool, or degrade with clear uncertainty."
    if base_error_type == ToolErrorType.PERMISSION_DENIED:
        return "Explain that the current credentials or permissions cannot access this tool."
    return "Retry only if safe, choose another tool, or degrade with clear uncertainty."


def _allowed_actions_for_error(
    base_error_type: ToolErrorType,
    retry_exhausted: bool,
) -> list[str]:
    if retry_exhausted:
        return [
            "use_alternative_tool",
            "search_memory_for_historical_reference",
            "search_rag_for_runbook",
            "degrade_with_unconfirmed_items",
        ]
    if base_error_type == ToolErrorType.PERMISSION_DENIED:
        return ["degrade_with_user_friendly_error"]
    if base_error_type == ToolErrorType.TOOL_TIMEOUT:
        return [
            "retry_same_tool_with_narrower_scope",
            "use_alternative_tool",
            "search_memory_for_historical_reference",
            "search_rag_for_runbook",
            "degrade_with_unconfirmed_items",
        ]
    return [
        "retry_same_tool_once_with_clearer_arguments",
        "use_alternative_tool",
        "search_memory_for_historical_reference",
        "search_rag_for_runbook",
        "degrade_with_unconfirmed_items",
    ]


def _disallowed_actions_for_error(retry_exhausted: bool) -> list[str]:
    actions = [
        "do_not_expose_raw_exception",
        "do_not_claim_root_cause_without_evidence",
    ]
    if retry_exhausted:
        actions.append("do_not_retry_after_budget_exhausted")
    return actions


def _controlled_allowed_actions(exception: BaseException) -> list[str] | None:
    actions = getattr(exception, "allowed_next_actions", None)
    if not isinstance(actions, (list, tuple)) or not actions:
        return None
    allowed = {
        "use_alternative_tool",
        "search_memory_for_historical_reference",
        "degrade_with_user_friendly_error",
        "ask_admin_to_configure_knowledge_base",
    }
    if any(not isinstance(action, str) or action not in allowed for action in actions):
        return None
    return list(actions)
