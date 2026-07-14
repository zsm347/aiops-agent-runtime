from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from superbiz_agent.tools.registry import ToolDefinition


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelMessage:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelStreamChunk:
    content_delta: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


class ModelGateway(Protocol):
    async def complete(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        ...

    def stream(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        ...
