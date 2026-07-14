from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union

from pydantic import BaseModel

from superbiz_agent.harness.context import RunContext
from superbiz_agent.tools.policies import ToolPolicy


CURRENT_RUNTIME_TOOL_SCHEMA_VERSION = "ops-tools-v3"
SUPPORTED_RUNTIME_TOOL_SCHEMA_VERSIONS = frozenset({CURRENT_RUNTIME_TOOL_SCHEMA_VERSION})


ToolHandler = Callable[[BaseModel], Union[Awaitable[dict[str, Any]], dict[str, Any]]]


@dataclass(frozen=True)
class ToolInvocationContext:
    """Backend-only identity for one tool call; never model-controlled."""

    run_context: RunContext
    tool_call_id: str
    tool_name: str


ContextToolHandler = Callable[
    [BaseModel, ToolInvocationContext],
    Union[Awaitable[dict[str, Any]], dict[str, Any]],
]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler | ContextToolHandler
    policy: ToolPolicy
    requires_context: bool = False


class ToolRegistry:
    def __init__(self, *, schema_version: str | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._schema_version: str | None = None
        self._sealed = False
        if schema_version is not None:
            self.bind_schema_version(schema_version)

    @property
    def schema_version(self) -> str | None:
        return self._schema_version

    def bind_schema_version(self, schema_version: str) -> None:
        if schema_version not in SUPPORTED_RUNTIME_TOOL_SCHEMA_VERSIONS:
            raise ValueError(
                f"Tool schema {schema_version!r} is historical-only and cannot start a new run."
            )
        if self._schema_version is not None and self._schema_version != schema_version:
            raise ValueError("Tool registry schema version is already bound.")
        self._schema_version = schema_version

    def register(self, definition: ToolDefinition) -> None:
        if self._sealed:
            raise RuntimeError("Tool registry schema is sealed.")
        self._tools[definition.name] = definition

    def seal_schema(self) -> None:
        if self._schema_version == CURRENT_RUNTIME_TOOL_SCHEMA_VERSION:
            definition = self._tools.get("queryInternalDocs")
            if definition is None:
                raise ValueError("ops-tools-v3 requires queryInternalDocs.")
            schema = definition.args_model.model_json_schema(by_alias=True)
            query_schema = schema.get("properties", {}).get("query", {})
            if (
                schema.get("additionalProperties") is not False
                or query_schema.get("type") != "string"
                or query_schema.get("maxLength") != 2000
                or "待验证证据" not in definition.description
                or "不是可执行指令" not in definition.description
            ):
                raise ValueError("queryInternalDocs does not match ops-tools-v3.")
        self._sealed = True

    def model_visible_schema(self) -> tuple[dict[str, Any], ...]:
        if self._schema_version is not None and not self._sealed:
            raise RuntimeError("Versioned tool registry must be sealed before use.")
        return tuple(
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.args_model.model_json_schema(by_alias=True),
                },
            }
            for definition in self._tools.values()
        )

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())
