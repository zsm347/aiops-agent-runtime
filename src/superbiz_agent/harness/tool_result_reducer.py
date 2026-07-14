from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from superbiz_agent.harness.content_compression import ContentCompressionBackend
from superbiz_agent.harness.context_budget import ContextBudget
from superbiz_agent.harness.context import RunContext
from superbiz_agent.harness.token_estimator import TokenEstimator
from superbiz_agent.model_gateway.base import ModelMessage, ModelToolCall
from superbiz_agent.security.redaction import (
    is_sensitive_key as _common_is_sensitive_key,
    redact_text as _common_redact_text,
    redact_value as _common_redact_value,
    sanitize_tool_result as _common_sanitize_tool_result,
)


@dataclass(frozen=True)
class ToolResultReduction:
    messages: list[ModelMessage]
    original_tool_result_count: int = 0
    reduced_tool_result_count: int = 0
    compressed_tool_result_count: int = 0
    cleared_tool_result_count: int = 0
    orphan_tool_result_count: int = 0
    raw_refs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    before_tokens: int = 0
    after_tokens: int = 0

    @property
    def changed(self) -> bool:
        return bool(
            self.reduced_tool_result_count
            or self.compressed_tool_result_count
            or self.cleared_tool_result_count
            or self.orphan_tool_result_count
        )

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "originalToolResultCount": self.original_tool_result_count,
            "reducedToolResultCount": self.reduced_tool_result_count,
            "compressedToolResultCount": self.compressed_tool_result_count,
            "clearedToolResultCount": self.cleared_tool_result_count,
            "orphanToolResultCount": self.orphan_tool_result_count,
            "rawRefs": self.raw_refs,
            "warnings": self.warnings,
            "estimatedTokensBefore": self.before_tokens,
            "estimatedTokensAfter": self.after_tokens,
        }


class ToolResultReducer:
    def __init__(
        self,
        *,
        budget: ContextBudget,
        estimator: TokenEstimator,
        compression_backend: ContentCompressionBackend,
    ) -> None:
        self.budget = budget
        self.estimator = estimator
        self.compression_backend = compression_backend

    def reduce(
        self,
        messages: list[ModelMessage],
        *,
        run_context: RunContext,
    ) -> ToolResultReduction:
        tool_indices = [index for index, message in enumerate(messages) if message.role == "tool"]
        keep_indices = set(tool_indices[-max(0, self.budget.tool_results_to_keep) :])
        result_ids = {
            message.tool_call_id
            for message in messages
            if message.role == "tool" and message.tool_call_id
        }

        reduced: list[ModelMessage] = []
        pending_tool_call_ids: set[str] = set()
        raw_refs: list[str] = []
        warnings: list[str] = []
        reduced_count = 0
        compressed_count = 0
        cleared_count = 0
        orphan_count = 0

        for index, message in enumerate(messages):
            if message.role == "assistant" and message.tool_calls:
                matched_tool_calls = [
                    tool_call for tool_call in message.tool_calls if tool_call.id in result_ids
                ]
                if not matched_tool_calls:
                    if message.content.strip():
                        reduced.append(_copy_message(message, tool_calls=[]))
                    continue
                pending_tool_call_ids.update(tool_call.id for tool_call in matched_tool_calls)
                reduced.append(_copy_message(message, tool_calls=matched_tool_calls))
                continue

            if message.role != "tool":
                reduced.append(message)
                continue

            if message.tool_call_id not in pending_tool_call_ids:
                orphan_count += 1
                warnings.append(
                    f"orphan tool result omitted: toolCallId={message.tool_call_id or 'missing'}"
                )
                continue

            pending_tool_call_ids.discard(message.tool_call_id or "")
            raw_ref = _raw_ref(run_context, message, index)
            raw_refs.append(raw_ref)
            before_tokens = self.estimator.estimate_text(message.content)
            if index not in keep_indices:
                cleared_count += 1
                reduced_count += 1
                reduced.append(
                    _copy_message(
                        message,
                        content=_placeholder(message, raw_ref, before_tokens),
                    )
                )
                continue

            sanitized = sanitize_tool_result(message.content)
            sanitized_text = _json_content(sanitized)
            if before_tokens > self.budget.tool_result_compress_threshold_tokens:
                compression = self.compression_backend.compress(
                    sanitized_text,
                    metadata={
                        "tool_name": message.name,
                        "tool_call_id": message.tool_call_id,
                        "raw_ref": raw_ref,
                    },
                )
                envelope = _compressed_envelope(
                    message=message,
                    raw_ref=raw_ref,
                    sanitized=sanitized,
                    summary=compression.content,
                    backend=compression.backend,
                    compressed=compression.compressed,
                    warnings=compression.warnings,
                    before_tokens=before_tokens,
                    after_tokens=self.estimator.estimate_text(compression.content),
                    threshold_tokens=self.budget.tool_result_compress_threshold_tokens,
                )
                warnings.extend(compression.warnings)
                compressed_count += 1
                reduced_count += 1
                reduced.append(_copy_message(message, content=_json_content(envelope)))
            else:
                if sanitized_text != message.content:
                    reduced_count += 1
                reduced.append(_copy_message(message, content=sanitized_text))

        if pending_tool_call_ids:
            warnings.append(
                "assistant tool calls without paired tool result were omitted from reducer input"
            )

        return ToolResultReduction(
            messages=reduced,
            original_tool_result_count=len(tool_indices),
            reduced_tool_result_count=reduced_count,
            compressed_tool_result_count=compressed_count,
            cleared_tool_result_count=cleared_count,
            orphan_tool_result_count=orphan_count,
            raw_refs=raw_refs,
            warnings=warnings,
            before_tokens=self.estimator.estimate_messages(messages),
            after_tokens=self.estimator.estimate_messages(reduced),
        )


def sanitize_tool_result(content: str) -> Any:
    return _common_sanitize_tool_result(content)


def redact_value(value: Any) -> Any:
    return _common_redact_value(value)


def redact_text(text: str) -> str:
    return _common_redact_text(text, replacement="[REDACTED]")


def _is_sensitive_key(key: str) -> bool:
    return _common_is_sensitive_key(key)


def _compressed_envelope(
    *,
    message: ModelMessage,
    raw_ref: str,
    sanitized: Any,
    summary: str,
    backend: str,
    compressed: bool,
    warnings: list[str],
    before_tokens: int,
    after_tokens: int,
    threshold_tokens: int,
) -> dict[str, Any]:
    return {
        "toolName": message.name,
        "toolCallId": message.tool_call_id,
        "status": _field(sanitized, "status"),
        "success": _field(sanitized, "success"),
        "errorType": _field(sanitized, "errorType") or _field(sanitized, "error_type"),
        "errorMessage": _field(sanitized, "errorMessage") or _field(sanitized, "message"),
        "resultCount": _result_count(sanitized),
        "keyFields": _key_fields(sanitized),
        "summary": summary,
        "raw_ref": raw_ref,
        "compression": {
            "compressed": compressed,
            "backend": backend,
            "beforeTokens": before_tokens,
            "afterTokens": after_tokens,
            "thresholdTokens": threshold_tokens,
            "warnings": warnings,
        },
    }


def _placeholder(message: ModelMessage, raw_ref: str, before_tokens: int) -> str:
    return (
        "[tool result cleared: "
        f"toolName={message.name or 'unknown'}, "
        f"toolCallId={message.tool_call_id or 'unknown'}, "
        f"originalTokens={before_tokens}, "
        f"raw_ref={raw_ref}, "
        "raw event retained in rollout store]"
    )


def _raw_ref(run_context: RunContext, message: ModelMessage, index: int) -> str:
    sequence = message.metadata.get("rollout_sequence")
    source_run_id = message.metadata.get("run_id") or run_context.run_id
    return (
        "rollout://"
        f"tenant/{run_context.request_context.tenant_id}/"
        f"session/{run_context.request_context.session_id}/"
        f"run/{source_run_id}/"
        f"tool_call/{message.tool_call_id or f'index-{index}'}/"
        f"sequence/{sequence if sequence is not None else 'unknown'}"
    )


def _json_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _result_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if not isinstance(value, dict):
        return None
    for key in ("resultCount", "count", "total"):
        item = value.get(key)
        if isinstance(item, int):
            return item
    for key in ("alerts", "logs", "chunks", "topics", "memories", "items", "results"):
        item = value.get(key)
        if isinstance(item, list):
            return len(item)
    return None


def _key_fields(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = (
        "success",
        "status",
        "message",
        "error_type",
        "errorType",
        "region",
        "logTopic",
        "query",
        "total",
        "count",
        "defaultRegion",
    )
    fields = {key: value[key] for key in keys if key in value}
    for list_key in ("alerts", "logs", "chunks", "topics", "memories", "results"):
        item = value.get(list_key)
        if isinstance(item, list):
            fields[f"{list_key}Count"] = len(item)
            fields[f"{list_key}Sample"] = item[:2]
    return fields


def _copy_message(
    message: ModelMessage,
    *,
    content: str | None = None,
    tool_calls: list[ModelToolCall] | None = None,
) -> ModelMessage:
    return ModelMessage(
        role=message.role,
        content=message.content if content is None else content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=message.tool_calls if tool_calls is None else tool_calls,
        metadata=dict(message.metadata),
    )
