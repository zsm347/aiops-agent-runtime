"""Small eval-only scripted model gateway for evaluator conformance fixtures.

It deliberately accepts only explicitly supplied responses.  Dataset expected
actions are never converted into scripts by this module.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable

from superbiz_agent.model_gateway.base import (
    ModelMessage,
    ModelResponse,
    ModelStreamChunk,
)
from superbiz_agent.tools.registry import ToolDefinition


class ScriptedModelGateway:
    """Return a finite, caller-defined sequence of model responses."""

    provider_name = "eval-scripted"

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = deque(responses)

    async def complete(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        del messages, tools
        if not self._responses:
            return ModelResponse(
                content="",
                finish_reason="script_exhausted",
                raw={"provider": self.provider_name},
            )
        response = self._responses.popleft()
        return ModelResponse(
            content=response.content,
            tool_calls=response.tool_calls,
            usage=response.usage,
            finish_reason=response.finish_reason,
            raw={**response.raw, "provider": self.provider_name},
        )

    def stream(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[ModelStreamChunk]:
        async def generate() -> AsyncIterator[ModelStreamChunk]:
            response = await self.complete(messages, tools)
            yield ModelStreamChunk(
                content_delta=response.content,
                tool_calls=response.tool_calls,
                usage=response.usage,
                finish_reason=response.finish_reason,
                raw=response.raw,
            )

        return generate()
