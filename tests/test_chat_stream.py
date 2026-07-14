from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from superbiz_agent.api.app import create_app
from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.graph import AgentStreamEvent
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.model_gateway.base import ModelMessage
from superbiz_agent.model_gateway.stub import StubModelGateway


def _parse_sse_messages(text: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for frame in text.strip().split("\n\n"):
        lines = frame.splitlines()
        if not lines:
            continue
        assert lines[0] == "event: message"
        data_lines = [line.removeprefix("data: ") for line in lines[1:] if line.startswith("data: ")]
        assert len(data_lines) == 1
        messages.append(json.loads(data_lines[0]))
    return messages


def _headers(session_tag: str) -> dict[str, str]:
    return {
        "X-Tenant-Id": "tenant-stream",
        "X-User-Id": "user-stream",
        "X-Agent-Id": "agent-stream",
        "X-Request-Id": f"request-{session_tag}",
    }


def _context(session_id: str) -> AgentRequestContext:
    return AgentRequestContext(
        "tenant-stream",
        "user-stream",
        "agent-stream",
        session_id,
    )


def test_chat_stream_empty_question_returns_error_done_and_no_run_events() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/chat_stream",
        json={"Id": "stream-empty", "Question": "  "},
        headers=_headers("empty"),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    messages = _parse_sse_messages(response.text)
    assert messages == [
        {"type": "error", "data": "问题内容不能为空"},
        {"type": "done", "data": None},
    ]

    events = asyncio.run(app.state.harness_service.trace_store.list_by_session(_context("stream-empty")))
    assert events == []


def test_chat_stream_plain_chat_emits_content_final_done() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/chat_stream",
        json={"Id": "stream-plain", "Question": "hello streaming"},
        headers=_headers("plain"),
    )

    assert response.status_code == 200
    messages = _parse_sse_messages(response.text)
    assert messages[0]["type"] == "content"
    assert [message["type"] for message in messages[-2:]] == ["final", "done"]
    content = "".join(message["data"] for message in messages if message["type"] == "content")
    assert content == "[stub] hello streaming"
    assert messages[-2]["data"] == {"data": "[stub] hello streaming"}
    assert messages[-1]["data"] is None

    events = asyncio.run(app.state.harness_service.trace_store.list_by_session(_context("stream-plain")))
    event_types = [event.event_type for event in events]
    assert RolloutEventType.RUN_COMPLETED in event_types
    assert RolloutEventType.RUN_FAILED not in event_types


def test_chat_stream_tool_question_preserves_trace() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/chat_stream",
        json={"Id": "stream-tool", "Question": "现在几点？"},
        headers=_headers("tool"),
    )

    assert response.status_code == 200
    messages = _parse_sse_messages(response.text)
    assert messages[-2]["type"] == "final"
    final_answer = messages[-2]["data"]["data"]
    assert "Asia/Shanghai" in final_answer
    assert "".join(message["data"] for message in messages if message["type"] == "content") == final_answer
    assert messages[-1] == {"type": "done", "data": None}

    events = asyncio.run(app.state.harness_service.trace_store.list_by_session(_context("stream-tool")))
    event_types = [event.event_type for event in events]
    assert RolloutEventType.TOOL_CALL_STARTED in event_types
    assert RolloutEventType.TOOL_CALL_COMPLETED in event_types
    assert event_types.count(RolloutEventType.MODEL_CALL_STARTED) == 2
    assert event_types.count(RolloutEventType.MODEL_CALL_COMPLETED) == 2
    assert event_types[-2:] == [
        RolloutEventType.ASSISTANT_MESSAGE_APPENDED,
        RolloutEventType.RUN_COMPLETED,
    ]


def test_chat_stream_failure_emits_error_done_and_records_run_failed() -> None:
    class FailingGraph:
        async def run_stream(
            self,
            run_context: Any,
            messages: list[ModelMessage],
        ) -> AsyncIterator[AgentStreamEvent]:
            raise RuntimeError("stream exploded")
            yield AgentStreamEvent(type="done")

    app = create_app()
    service = AgentHarnessService.placeholder()
    service.graph = FailingGraph()  # type: ignore[assignment]
    app.state.harness_service = service
    client = TestClient(app)

    response = client.post(
        "/api/chat_stream",
        json={"Id": "stream-fail", "Question": "hello"},
        headers=_headers("fail"),
    )

    assert response.status_code == 200
    messages = _parse_sse_messages(response.text)
    assert messages == [
        {"type": "error", "data": "stream exploded"},
        {"type": "done", "data": None},
    ]

    events = asyncio.run(service.trace_store.list_by_session(_context("stream-fail")))
    event_types = [event.event_type for event in events]
    assert RolloutEventType.RUN_FAILED in event_types
    assert RolloutEventType.RUN_COMPLETED not in event_types


@pytest.mark.asyncio
async def test_stub_model_gateway_stream_chunks() -> None:
    gateway = StubModelGateway()

    chunks = [
        chunk
        async for chunk in gateway.stream(
            [ModelMessage(role="user", content="hello streaming chunks")]
        )
    ]

    content = "".join(chunk.content_delta for chunk in chunks)
    assert content == "[stub] hello streaming chunks"
    assert [chunk.content_delta for chunk in chunks[:-1]] == [
        "[stub] hello str",
        "eaming chunks",
    ]
    assert chunks[-1].content_delta == ""
    assert chunks[-1].raw["mode"] == "deterministic_answer"

