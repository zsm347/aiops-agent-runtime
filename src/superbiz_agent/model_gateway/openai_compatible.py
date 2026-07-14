from __future__ import annotations

import asyncio
import json
from typing import Any

from superbiz_agent.model_gateway.base import (
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from superbiz_agent.model_gateway.errors import (
    ModelGatewayAuthError,
    ModelGatewayContextOverflowError,
    ModelGatewayError,
    ModelGatewayProviderError,
    ModelGatewayRateLimitError,
    ModelGatewayTimeoutError,
)
from superbiz_agent.tools.registry import ToolDefinition


class OpenAICompatibleModelGateway:
    """Thin OpenAI SDK adapter for OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str | None = None,
        timeout_ms: int = 180_000,
        max_retries: int = 0,
        client: Any | None = None,
        provider_name: str = "openai-compatible",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.timeout_ms = timeout_ms
        self.max_retries = max(0, max_retries)
        self.provider_name = provider_name
        self._api_key = api_key
        self._client = client

    async def complete(
        self,
        messages: list[ModelMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [_message_to_openai(message) for message in messages],
            "timeout": self.timeout_ms / 1000,
        }
        if tools:
            request["tools"] = [_tool_to_openai(tool) for tool in tools]
            request["tool_choice"] = "auto"

        client = self._client
        if client is None:
            client = self._build_client(self._api_key, self.base_url, self.timeout_ms)
            self._client = client

        try:
            response = await client.chat.completions.create(**request)
        except Exception as exc:
            raise self._normalize_error(exc) from exc

        return _parse_response(response, self.provider_name)

    @staticmethod
    def _build_client(api_key: str, base_url: str | None, timeout_ms: int) -> Any:
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env.
            raise RuntimeError("openai package is required for real model providers") from exc

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_ms / 1000,
            "max_retries": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return AsyncOpenAI(**kwargs)

    def _normalize_error(self, exc: Exception) -> ModelGatewayError:
        message = _error_message(exc)
        status_code = getattr(exc, "status_code", None)
        class_name = exc.__class__.__name__

        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or class_name == "APITimeoutError":
            return ModelGatewayTimeoutError(message, provider=self.provider_name)
        if class_name == "AuthenticationError":
            return ModelGatewayAuthError(
                message,
                provider=self.provider_name,
                status_code=status_code,
            )
        if class_name == "RateLimitError":
            return ModelGatewayRateLimitError(
                message,
                provider=self.provider_name,
                status_code=status_code,
            )
        if _looks_like_context_overflow(message):
            return ModelGatewayContextOverflowError(
                message,
                provider=self.provider_name,
                status_code=status_code,
            )
        if class_name == "BadRequestError":
            return ModelGatewayProviderError(
                message,
                provider=self.provider_name,
                status_code=status_code,
            )
        if class_name == "APIStatusError" or isinstance(status_code, int):
            if status_code in (401, 403):
                return ModelGatewayAuthError(
                    message,
                    provider=self.provider_name,
                    status_code=status_code,
                )
            if status_code == 429:
                return ModelGatewayRateLimitError(
                    message,
                    provider=self.provider_name,
                    status_code=status_code,
                )
            if status_code == 408:
                return ModelGatewayTimeoutError(
                    message,
                    provider=self.provider_name,
                    status_code=status_code,
                )
            return ModelGatewayProviderError(
                message,
                provider=self.provider_name,
                status_code=status_code,
                retryable=bool(status_code and status_code >= 500),
            )
        if class_name == "APIConnectionError":
            return ModelGatewayProviderError(
                message,
                provider=self.provider_name,
                retryable=True,
            )
        return ModelGatewayProviderError(
            message,
            provider=self.provider_name,
            status_code=status_code,
        )


def _message_to_openai(message: ModelMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.name:
        payload["name"] = message.name
    if message.role == "tool" and message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        tool_call.arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _tool_to_openai(definition: ToolDefinition) -> dict[str, Any]:
    schema = definition.args_model.model_json_schema(by_alias=True)
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": schema,
        },
    }


def _parse_response(response: Any, provider_name: str) -> ModelResponse:
    choice = _first(_read(response, "choices") or [])
    message = _read(choice, "message") if choice is not None else None
    content = _content_to_text(_read(message, "content"))
    parse_errors: list[dict[str, str]] = []
    tool_calls = _parse_tool_calls(_read(message, "tool_calls") or [], parse_errors)
    usage = _parse_usage(_read(response, "usage"))
    finish_reason = _read(choice, "finish_reason")
    raw = {
        "provider": provider_name,
        "id": _read(response, "id"),
        "model": _read(response, "model"),
        "finish_reason": finish_reason,
        "usage": usage,
        "tool_call_count": len(tool_calls),
    }
    if parse_errors:
        raw["tool_call_parse_errors"] = parse_errors
    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish_reason,
        raw=raw,
    )


def _parse_tool_calls(tool_calls: list[Any], parse_errors: list[dict[str, str]]) -> list[ModelToolCall]:
    parsed: list[ModelToolCall] = []
    for index, tool_call in enumerate(tool_calls):
        function = _read(tool_call, "function")
        raw_id = _read(tool_call, "id")
        tool_call_id = raw_id if isinstance(raw_id, str) else ""
        if not tool_call_id.strip():
            parse_errors.append({"index": str(index), "reason": "missing_tool_call_id"})
        raw_name = _read(function, "name")
        name = raw_name if isinstance(raw_name, str) else ""
        if not name.strip():
            parse_errors.append({"index": str(index), "reason": "invalid_tool_name"})
        raw_arguments = _read(function, "arguments")
        arguments: dict[str, Any] = {}
        if isinstance(raw_arguments, str) and raw_arguments.strip():
            try:
                loaded = json.loads(raw_arguments)
                if isinstance(loaded, dict):
                    arguments = loaded
                else:
                    parse_errors.append(
                        {"index": str(index), "reason": "arguments_not_object"}
                    )
            except json.JSONDecodeError:
                parse_errors.append({"index": str(index), "reason": "invalid_json_arguments"})
        parsed.append(
            ModelToolCall(
                id=tool_call_id,
                name=name,
                arguments=arguments,
            )
        )
    return parsed


def _parse_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    key_map = {
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
    }
    parsed: dict[str, int] = {}
    for raw_key, output_key in key_map.items():
        value = _read(usage, raw_key)
        if isinstance(value, int):
            parsed[output_key] = value
    return parsed


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _read(item, "text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


def _read(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _first(values: Any) -> Any:
    if isinstance(values, (list, tuple)) and values:
        return values[0]
    return None


def _error_message(exc: Exception) -> str:
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(exc)


def _looks_like_context_overflow(message: str) -> bool:
    lowered = message.lower()
    return any(
        keyword in lowered
        for keyword in (
            "context length",
            "context_length",
            "maximum context",
            "max context",
            "too many tokens",
            "token limit",
            "input is too long",
        )
    )
