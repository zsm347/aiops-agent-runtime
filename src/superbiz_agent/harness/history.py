from __future__ import annotations

import json
from collections.abc import Sequence

from superbiz_agent.harness.events import RolloutEvent, RolloutEventType
from superbiz_agent.model_gateway.base import ModelMessage, ModelToolCall


def recover_active_history(events: Sequence[RolloutEvent]) -> list[ModelMessage]:
    events_to_replay = list(events)
    history: list[ModelMessage] = []
    latest_trimmed = _latest_successful_history_trimmed(events_to_replay)
    if latest_trimmed is not None:
        payload = latest_trimmed.payload or {}
        summary = payload.get("summary")
        if isinstance(summary, str) and summary.strip():
            history.append(
                ModelMessage(
                    role="system",
                    content=summary,
                    metadata={
                        "context_summary": True,
                        "rollout_sequence": latest_trimmed.sequence,
                        "event_id": latest_trimmed.event_id,
                        "run_id": latest_trimmed.run_id,
                    },
                )
            )
        boundary_sequence = _history_trim_boundary(latest_trimmed)
        events_to_replay = [
            event
            for event in events_to_replay
            if event.sequence > boundary_sequence
            and event.event_type != RolloutEventType.HISTORY_TRIMMED
        ]

    pending_tool_calls: dict[str, tuple[ModelToolCall, RolloutEvent]] = {}
    for event in events_to_replay:
        payload = event.payload or {}
        if event.event_type == RolloutEventType.USER_MESSAGE_APPENDED:
            content = payload.get("content")
            if isinstance(content, str):
                history.append(
                    ModelMessage(role="user", content=content, metadata=_event_metadata(event))
                )
        elif event.event_type == RolloutEventType.ASSISTANT_MESSAGE_APPENDED:
            content = payload.get("content")
            if isinstance(content, str):
                history.append(
                    ModelMessage(
                        role="assistant",
                        content=content,
                        metadata=_event_metadata(event),
                    )
                )
        elif event.event_type == RolloutEventType.TOOL_CALL_STARTED:
            tool_name = payload.get("toolName")
            arguments = payload.get("arguments")
            if isinstance(tool_name, str) and tool_name:
                tool_call_id = event.tool_call_id or f"toolcall-{event.sequence}"
                pending_tool_calls[tool_call_id] = (
                    ModelToolCall(
                        id=tool_call_id,
                        name=tool_name,
                        arguments=arguments if isinstance(arguments, dict) else {},
                    ),
                    event,
                )
        elif event.event_type in {
            RolloutEventType.TOOL_CALL_COMPLETED,
            RolloutEventType.TOOL_CALL_FAILED,
            RolloutEventType.TOOL_CALL_BLOCKED,
        }:
            tool_name = payload.get("toolName")
            result = payload.get("result")
            if isinstance(tool_name, str) and tool_name:
                tool_call_id = event.tool_call_id or f"toolcall-{event.sequence}"
                tool_call, start_event = pending_tool_calls.pop(
                    tool_call_id,
                    (
                        ModelToolCall(id=tool_call_id, name=tool_name, arguments={}),
                        event,
                    ),
                )
                history.append(
                    ModelMessage(
                        role="assistant",
                        content="",
                        tool_calls=[tool_call],
                        metadata=_event_metadata(start_event),
                    )
                )
                history.append(
                    ModelMessage(
                        role="tool",
                        name=tool_name,
                        content=_json_content(result),
                        tool_call_id=tool_call_id,
                        metadata=_event_metadata(event),
                    )
                )
    return history


def count_message_pairs(events: Sequence[RolloutEvent]) -> int:
    user_count = sum(
        1 for event in events if event.event_type == RolloutEventType.USER_MESSAGE_APPENDED
    )
    assistant_count = sum(
        1 for event in events if event.event_type == RolloutEventType.ASSISTANT_MESSAGE_APPENDED
    )
    return min(user_count, assistant_count)


def _json_content(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _latest_successful_history_trimmed(
    events: Sequence[RolloutEvent],
) -> RolloutEvent | None:
    candidates = [
        event
        for event in events
        if event.event_type == RolloutEventType.HISTORY_TRIMMED
        and (event.payload or {}).get("status") == "success"
        and isinstance((event.payload or {}).get("summary"), str)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda event: event.sequence)


def _history_trim_boundary(event: RolloutEvent) -> int:
    to_sequence = (event.payload or {}).get("toSequence")
    if isinstance(to_sequence, int):
        return to_sequence
    return event.sequence


def _event_metadata(event: RolloutEvent) -> dict:
    return {
        "rollout_sequence": event.sequence,
        "event_id": event.event_id,
        "run_id": event.run_id,
        "event_type": event.event_type.value,
    }
