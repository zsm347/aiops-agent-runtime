"""Eval-only model gateway capture with shared redaction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any, Literal

from superbiz_agent.evals.memory_artifacts import (
    ModelCallArtifact,
    ModelMessageCapture,
    ToolSchemaCapture,
)
from superbiz_agent.model_gateway.base import (
    ModelGateway,
    ModelMessage,
    ModelResponse,
    ModelStreamChunk,
)
from superbiz_agent.security.redaction import redact_text, redact_value, sanitize_error_message
from superbiz_agent.tools.registry import ToolDefinition

_MAX_CAPTURE_TEXT = 800


class CapturingModelGateway:
    """Transparent gateway wrapper that keeps redacted calls in eval memory only."""

    def __init__(self, wrapped: ModelGateway) -> None:
        self.wrapped = wrapped
        self._calls: list[ModelCallArtifact] = []

    @property
    def captured_calls(self) -> tuple[ModelCallArtifact, ...]:
        return tuple(self._calls)

    def drain(self) -> list[ModelCallArtifact]:
        calls = list(self._calls)
        self._calls.clear()
        return calls

    async def complete(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        call = self._new_call("complete", messages, tools)
        try:
            response = await self.wrapped.complete(messages, tools=tools)
        except Exception as exc:
            call.error = sanitize_error_message(str(exc))
            self._calls.append(call)
            raise
        self._record_response(call, response)
        self._calls.append(call)
        return response

    def stream(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        async def captured_stream() -> AsyncIterator[ModelStreamChunk]:
            call = self._new_call("stream", messages, tools)
            content_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            usage: dict[str, int] = {}
            raw_metadata: dict[str, Any] = {}
            finish_reason: str | None = None
            try:
                async for chunk in self.wrapped.stream(messages, tools=tools):
                    content_parts.append(chunk.content_delta)
                    tool_calls.extend(_tool_calls_payload(chunk.tool_calls))
                    usage.update(chunk.usage)
                    raw_metadata.update(redact_value(chunk.raw))
                    finish_reason = chunk.finish_reason or finish_reason
                    yield chunk
            except Exception as exc:
                call.error = sanitize_error_message(str(exc))
                raise
            finally:
                call.response_content = _summary("".join(content_parts))
                call.tool_calls = tool_calls
                call.usage = usage
                call.raw_metadata = raw_metadata
                call.finish_reason = finish_reason
                self._calls.append(call)

        return captured_stream()

    def _new_call(
        self,
        mode: Literal["complete", "stream"],
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None,
    ) -> ModelCallArtifact:
        return ModelCallArtifact(
            call_index=len(self._calls) + 1,
            mode=mode,
            messages=[_capture_message(message) for message in messages],
            tool_schemas=[_capture_tool_schema(tool) for tool in tools or []],
        )

    @staticmethod
    def _record_response(call: ModelCallArtifact, response: ModelResponse) -> None:
        call.response_content = _summary(response.content)
        call.tool_calls = _tool_calls_payload(response.tool_calls)
        call.usage = dict(response.usage)
        call.finish_reason = response.finish_reason
        call.raw_metadata = redact_value(response.raw)


def _capture_message(message: ModelMessage) -> ModelMessageCapture:
    return ModelMessageCapture(
        role=message.role,
        content_summary=_summary(message.content),
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_call_names=[call.name for call in message.tool_calls],
        metadata=redact_value(message.metadata),
    )


def _capture_tool_schema(tool: ToolDefinition) -> ToolSchemaCapture:
    schema = tool.args_model.model_json_schema(by_alias=True)
    canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    properties = schema.get("properties")
    keys = sorted(properties) if isinstance(properties, dict) else []
    return ToolSchemaCapture(
        name=tool.name,
        schema_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        description_summary=_summary(tool.description),
        argument_keys=keys,
    )


def _tool_calls_payload(tool_calls: list[Any]) -> list[dict[str, Any]]:
    return [
        {"id": call.id, "name": call.name, "arguments": redact_value(call.arguments)}
        for call in tool_calls
    ]


def _summary(value: str) -> str:
    redacted = redact_text(str(value), replacement="[REDACTED]")
    return redacted if len(redacted) <= _MAX_CAPTURE_TEXT else f"{redacted[:_MAX_CAPTURE_TEXT - 3]}..."
