from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.events import RolloutEvent, RolloutEventType
from superbiz_agent.harness.service import AgentChatResult


class TraceArtifact(BaseModel):
    """Eval-facing summary of a harness rollout trace."""

    model_config = ConfigDict(protected_namespaces=())

    run_id: Optional[str] = None
    session_id: str
    prompt_version: Optional[str] = None
    model_provider: Optional[str] = None
    tool_schema_version: Optional[str] = None
    event_types: list[str] = Field(default_factory=list)
    event_sequences: list[int] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    tool_arguments: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    tool_results: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    final_answer: Optional[str] = None
    success: bool
    error_message: Optional[str] = None
    latency_ms: Optional[int] = None
    raw_event_count: int = 0
    token_usage: Optional[dict[str, Any]] = None
    model_event_count: int = 0
    max_context_history_item_count: int = 0
    max_context_core_memory_block_count: int = 0
    max_context_memory_index_topic_count: int = 0
    saw_core_memory: bool = False
    saw_memory_index: bool = False
    saw_memory_metadata: bool = False

    @classmethod
    def from_events(
        cls,
        *,
        context: AgentRequestContext,
        chat_result: AgentChatResult,
        events: Sequence[RolloutEvent],
        latency_ms: Optional[int],
    ) -> "TraceArtifact":
        run_started = next(
            (event for event in events if event.event_type == RolloutEventType.RUN_STARTED),
            None,
        )
        prompt_version = None
        model_provider = None
        tool_schema_version = None
        if run_started is not None:
            prompt_version = run_started.payload.get("promptVersion")
            model_provider = run_started.payload.get("modelProvider")
            tool_schema_version = run_started.payload.get("toolSchemaVersion")

        tool_calls: list[str] = []
        tool_arguments: dict[str, list[dict[str, Any]]] = {}
        tool_results: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            tool_name = event.payload.get("toolName")
            if not isinstance(tool_name, str) or not tool_name:
                continue

            if tool_name not in tool_calls:
                tool_calls.append(tool_name)

            arguments = event.payload.get("arguments")
            if event.event_type == RolloutEventType.TOOL_CALL_STARTED and isinstance(
                arguments, dict
            ):
                tool_arguments.setdefault(tool_name, []).append(arguments)

            result = event.payload.get("result")
            if event.event_type in {
                RolloutEventType.TOOL_CALL_COMPLETED,
                RolloutEventType.TOOL_CALL_FAILED,
                RolloutEventType.TOOL_CALL_BLOCKED,
            } and isinstance(result, dict):
                tool_results.setdefault(tool_name, []).append(result)

        event_types = [event.event_type.value for event in events]
        model_event_count = sum(
            1 for event_type in event_types if event_type.startswith("MODEL_CALL_")
        )
        max_context_history_item_count = max(
            (
                event.payload.get("historyItemCount", 0)
                for event in events
                if event.event_type == RolloutEventType.CONTEXT_ASSEMBLED
                and isinstance(event.payload.get("historyItemCount", 0), int)
            ),
            default=0,
        )
        max_context_core_memory_block_count = max(
            (
                event.payload.get("coreMemoryBlockCount", 0)
                for event in events
                if event.event_type == RolloutEventType.CONTEXT_ASSEMBLED
                and isinstance(event.payload.get("coreMemoryBlockCount", 0), int)
            ),
            default=0,
        )
        max_context_memory_index_topic_count = max(
            (
                event.payload.get("memoryIndexTopicCount", 0)
                for event in events
                if event.event_type == RolloutEventType.CONTEXT_ASSEMBLED
                and isinstance(event.payload.get("memoryIndexTopicCount", 0), int)
            ),
            default=0,
        )
        saw_core_memory = any(
            bool(event.payload.get("hasCoreMemory"))
            for event in events
            if event.event_type == RolloutEventType.CONTEXT_ASSEMBLED
        )
        saw_memory_index = any(
            bool(event.payload.get("hasMemoryIndex"))
            for event in events
            if event.event_type == RolloutEventType.CONTEXT_ASSEMBLED
        )
        saw_memory_metadata = any(
            bool(event.payload.get("hasMemoryMetadata"))
            for event in events
            if event.event_type == RolloutEventType.CONTEXT_ASSEMBLED
        )
        run_id = chat_result.run_id or next((event.run_id for event in events if event.run_id), None)

        return cls(
            run_id=run_id,
            session_id=chat_result.session_id or context.session_id or "",
            prompt_version=prompt_version,
            model_provider=model_provider,
            tool_schema_version=tool_schema_version,
            event_types=event_types,
            event_sequences=[event.sequence for event in events],
            tool_calls=tool_calls,
            tool_arguments=tool_arguments,
            tool_results=tool_results,
            final_answer=chat_result.answer,
            success=chat_result.success,
            error_message=chat_result.error_message,
            latency_ms=latency_ms,
            raw_event_count=len(events),
            model_event_count=model_event_count,
            max_context_history_item_count=max_context_history_item_count,
            max_context_core_memory_block_count=max_context_core_memory_block_count,
            max_context_memory_index_topic_count=max_context_memory_index_topic_count,
            saw_core_memory=saw_core_memory,
            saw_memory_index=saw_memory_index,
            saw_memory_metadata=saw_memory_metadata,
        )
