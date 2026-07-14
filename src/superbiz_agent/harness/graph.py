from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from superbiz_agent.harness.context import RunContext
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.stores import RolloutEventStore
from superbiz_agent.harness.tool_result_reducer import ToolResultReducer
from superbiz_agent.model_gateway.base import (
    ModelGateway,
    ModelMessage,
    ModelResponse,
    ModelToolCall,
)
from superbiz_agent.model_gateway.errors import (
    ModelGatewayContextOverflowError,
    ModelGatewayError,
    ModelGatewayTimeoutError,
)
from superbiz_agent.tools.gateway import ToolExecutionResult, ToolGateway


_BUDGET_NEXT_ACTIONS = [
    "answer_with_available_evidence",
    "state_unverified_gaps",
    "ask_user_to_start_a_narrower_follow_up",
]
_PROTOCOL_NOTICE = (
    "The previous model tool-call batch was rejected because its protocol fields were invalid. "
    "No tool in that batch was executed. Answer using only available evidence and do not request "
    "more tools."
)
_EMPTY_ANSWER_FALLBACK = (
    "抱歉，本次请求未能生成可靠答案。请缩小问题范围后重试；未获得的工具证据不会被编造。"
)
_UNEXECUTED_FINAL_TOOL_NOTICE = "\n\n注意：收尾阶段请求的工具未执行，相关信息仍未确认。"


@dataclass(frozen=True)
class AgentStreamEvent:
    type: str
    data: Any = None


class SkeletonGraphState(TypedDict):
    run_context: RunContext
    messages: list[ModelMessage]
    final_answer: Optional[str]
    model_response: Optional[ModelResponse]
    tool_round_count: int
    tool_call_count: int
    seen_tool_call_ids: set[str]
    force_final_without_tools: bool
    tool_budget_exhausted: bool
    tool_result_reduction: Optional[dict[str, Any]]


class SkeletonAgentGraph:
    def __init__(
        self,
        *,
        model_gateway: ModelGateway,
        tool_gateway: ToolGateway,
        trace_store: RolloutEventStore,
        tool_result_reducer: ToolResultReducer | None = None,
        max_tool_rounds: int = 4,
        max_tool_calls_per_run: int = 8,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if max_tool_calls_per_run < 1:
            raise ValueError("max_tool_calls_per_run must be at least 1")
        self.model_gateway = model_gateway
        self.tool_gateway = tool_gateway
        self.trace_store = trace_store
        self.tool_result_reducer = tool_result_reducer
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls_per_run = max_tool_calls_per_run
        self.recursion_limit = 2 * max_tool_rounds + 6

        builder = StateGraph(SkeletonGraphState)
        builder.add_node("model_call", self._model_call)
        builder.add_node("tool_dispatch", self._tool_dispatch)
        builder.set_entry_point("model_call")
        builder.add_conditional_edges(
            "model_call",
            self._next_after_model_call,
            {"model_call": "model_call", "tool_dispatch": "tool_dispatch", "final": END},
        )
        builder.add_edge("tool_dispatch", "model_call")
        self._graph = builder.compile()

    async def run(self, run_context: RunContext, messages: list[ModelMessage]) -> str:
        state: SkeletonGraphState = {
            "run_context": run_context,
            "messages": messages,
            "final_answer": None,
            "model_response": None,
            "tool_round_count": 0,
            "tool_call_count": 0,
            "seen_tool_call_ids": set(),
            "force_final_without_tools": False,
            "tool_budget_exhausted": False,
            "tool_result_reduction": None,
        }
        try:
            result = await self._graph.ainvoke(
                state,
                config={"recursion_limit": self.recursion_limit},
            )
            return result.get("final_answer") or _EMPTY_ANSWER_FALLBACK
        finally:
            self.cleanup_run(run_context.run_id)

    def cleanup_run(self, run_id: str) -> None:
        self.tool_gateway.clear_run_state(run_id)

    async def run_stream(
        self,
        run_context: RunContext,
        messages: list[ModelMessage],
    ) -> AsyncIterator[AgentStreamEvent]:
        answer = await self.run(run_context, messages)
        for chunk in _chunk_text(answer):
            yield AgentStreamEvent(type="content", data=chunk)
        yield AgentStreamEvent(type="final", data={"data": answer})

    async def _model_call(self, state: SkeletonGraphState) -> dict[str, Any]:
        run_context = state["run_context"]
        messages = state["messages"]
        final_call = state["force_final_without_tools"]
        started_payload: dict[str, Any] = {
            "provider": run_context.model_provider,
            "messageCount": len(messages),
            "toolsEnabled": not final_call,
        }
        if state.get("tool_result_reduction"):
            started_payload["toolResultReduction"] = state["tool_result_reduction"]
        await self.trace_store.append_event(
            run_context,
            RolloutEventType.MODEL_CALL_STARTED,
            started_payload,
        )
        try:
            response = await self.model_gateway.complete(
                messages,
                tools=[] if final_call else self.tool_gateway.registry.list(),
            )
        except ModelGatewayTimeoutError as exc:
            await self._record_model_failure(
                run_context,
                exc,
                specific_event_type=RolloutEventType.MODEL_CALL_TIMEOUT,
            )
            raise
        except ModelGatewayContextOverflowError as exc:
            await self._record_model_failure(
                run_context,
                exc,
                specific_event_type=RolloutEventType.MODEL_CALL_CONTEXT_OVERFLOW,
            )
            raise
        except ModelGatewayError as exc:
            await self._record_model_failure(run_context, exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive path for future gateways.
            await self._record_model_failure(run_context, exc)
            raise

        await self.trace_store.append_event(
            run_context,
            RolloutEventType.MODEL_CALL_COMPLETED,
            {
                "provider": run_context.model_provider,
                "toolCallCount": len(response.tool_calls),
                "hasContent": bool(response.content.strip()),
                "finishReason": response.finish_reason,
                "toolsEnabled": not final_call,
            },
        )
        if response.usage:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.TOKEN_USAGE,
                {"provider": run_context.model_provider, "usage": response.usage},
            )

        update: dict[str, Any] = {
            "model_response": response,
            "tool_result_reduction": None,
        }
        if final_call:
            update["final_answer"] = await self._finalize_disabled_tool_response(
                run_context,
                response,
            )
            return update

        if response.tool_calls:
            violations = self._protocol_violations(
                response.tool_calls,
                state["seen_tool_call_ids"],
            )
            if violations:
                await self.trace_store.append_event(
                    run_context,
                    RolloutEventType.AGENT_MODEL_PROTOCOL_ERROR,
                    {
                        "status": "rejected",
                        "toolCallCount": len(response.tool_calls),
                        "violations": violations,
                    },
                )
                update.update(
                    {
                        "messages": [*messages, ModelMessage(role="system", content=_PROTOCOL_NOTICE)],
                        "model_response": None,
                        "force_final_without_tools": True,
                    }
                )
            return update

        if response.content.strip():
            update["final_answer"] = response.content
        else:
            update["final_answer"] = await self._record_empty_answer_fallback(run_context)
        return update

    async def _tool_dispatch(self, state: SkeletonGraphState) -> dict[str, Any]:
        response = state["model_response"]
        if response is None or not response.tool_calls:
            return {}

        tool_calls = response.tool_calls
        requested_round = state["tool_round_count"] + 1
        requested_call_count = state["tool_call_count"] + len(tool_calls)
        budget_reason = None
        if requested_round > self.max_tool_rounds:
            budget_reason = "agent_tool_round_budget_exhausted"
        elif requested_call_count > self.max_tool_calls_per_run:
            budget_reason = "agent_tool_call_budget_exhausted"

        messages = [
            *state["messages"],
            ModelMessage(
                role="assistant",
                content=response.content,
                tool_calls=tool_calls,
            ),
        ]
        results: list[ToolExecutionResult] = []
        if budget_reason is not None:
            await self.trace_store.append_event(
                state["run_context"],
                RolloutEventType.AGENT_TOOL_BUDGET_EXHAUSTED,
                {
                    "toolRoundCount": requested_round,
                    "toolCallCount": requested_call_count,
                    "maxToolRounds": self.max_tool_rounds,
                    "maxToolCallsPerRun": self.max_tool_calls_per_run,
                    "blockedToolCallCount": len(tool_calls),
                    "reason": budget_reason,
                    "status": "exhausted",
                },
            )
            for tool_call in tool_calls:
                results.append(
                    await self.tool_gateway.block_tool_call(
                        state["run_context"],
                        tool_call,
                        reason="agent_tool_budget_exhausted",
                        message=(
                            "The Agent run tool budget is exhausted; this tool call was not "
                            "executed."
                        ),
                        allowed_next_actions=_BUDGET_NEXT_ACTIONS,
                    )
                )
        else:
            for tool_call in tool_calls:
                results.append(await self.tool_gateway.execute(state["run_context"], tool_call))

        for result in results:
            messages.append(
                ModelMessage(
                    role="tool",
                    content=json.dumps(result.result, ensure_ascii=False, sort_keys=True),
                    name=result.tool_name,
                    tool_call_id=result.tool_call_id,
                    metadata=await self._tool_result_metadata(
                        state["run_context"],
                        result.tool_call_id,
                    ),
                )
            )

        reduction_payload = None
        if self.tool_result_reducer is not None:
            reduction = self.tool_result_reducer.reduce(
                messages,
                run_context=state["run_context"],
            )
            messages = reduction.messages
            if reduction.changed or reduction.warnings:
                reduction_payload = reduction.to_trace_payload()

        return {
            "messages": messages,
            "model_response": None,
            "tool_round_count": requested_round,
            "tool_call_count": requested_call_count,
            "seen_tool_call_ids": state["seen_tool_call_ids"]
            | {tool_call.id for tool_call in tool_calls},
            "force_final_without_tools": budget_reason is not None,
            "tool_budget_exhausted": budget_reason is not None,
            "tool_result_reduction": reduction_payload,
        }

    async def _tool_result_metadata(
        self,
        run_context: RunContext,
        tool_call_id: str,
    ) -> dict[str, Any]:
        try:
            events = await self.trace_store.list_by_run(
                run_context.request_context.tenant_id or "",
                run_context.run_id,
            )
        except Exception:  # pragma: no cover - metadata is best-effort only.
            return {}
        for event in reversed(events):
            if event.tool_call_id != tool_call_id:
                continue
            if event.event_type not in {
                RolloutEventType.TOOL_CALL_COMPLETED,
                RolloutEventType.TOOL_CALL_FAILED,
                RolloutEventType.TOOL_CALL_BLOCKED,
            }:
                continue
            return {
                "rollout_sequence": event.sequence,
                "event_id": event.event_id,
                "run_id": event.run_id,
                "event_type": event.event_type.value,
            }
        return {}

    @staticmethod
    def _next_after_model_call(state: SkeletonGraphState) -> str:
        if state.get("final_answer") is not None:
            return "final"
        if state.get("force_final_without_tools") and state.get("model_response") is None:
            return "model_call"
        response = state.get("model_response")
        if response is not None and response.tool_calls:
            return "tool_dispatch"
        return "final"

    @staticmethod
    def _protocol_violations(
        tool_calls: list[ModelToolCall],
        seen_tool_call_ids: set[str],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        batch_ids: set[str] = set()
        for index, tool_call in enumerate(tool_calls):
            if not tool_call.id.strip():
                violations.append({"index": index, "reason": "blank_tool_call_id"})
            elif tool_call.id in batch_ids:
                violations.append({"index": index, "reason": "duplicate_tool_call_id_in_batch"})
            elif tool_call.id in seen_tool_call_ids:
                violations.append({"index": index, "reason": "reused_tool_call_id_in_run"})
            if not tool_call.name.strip():
                violations.append({"index": index, "reason": "blank_tool_name"})
            batch_ids.add(tool_call.id)
        return violations

    async def _finalize_disabled_tool_response(
        self,
        run_context: RunContext,
        response: ModelResponse,
    ) -> str:
        if response.tool_calls:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.AGENT_FINALIZATION_FALLBACK,
                {
                    "reason": "tool_call_requested_while_tools_disabled",
                    "requestedToolCallCount": len(response.tool_calls),
                    "status": "fallback",
                },
            )
            if response.content.strip():
                return response.content + _UNEXECUTED_FINAL_TOOL_NOTICE
            return _EMPTY_ANSWER_FALLBACK
        if response.content.strip():
            return response.content
        return await self._record_empty_answer_fallback(run_context)

    async def _record_empty_answer_fallback(self, run_context: RunContext) -> str:
        await self.trace_store.append_event(
            run_context,
            RolloutEventType.AGENT_FINALIZATION_FALLBACK,
            {"reason": "empty_model_answer", "status": "fallback"},
        )
        return _EMPTY_ANSWER_FALLBACK

    async def _record_model_failure(
        self,
        run_context: RunContext,
        exc: Exception,
        *,
        specific_event_type: RolloutEventType | None = None,
    ) -> None:
        payload = {
            "provider": run_context.model_provider,
            "errorMessage": str(exc),
            "errorType": getattr(exc, "error_type", "provider-error"),
            "statusCode": getattr(exc, "status_code", None),
            "retryable": bool(getattr(exc, "retryable", False)),
        }
        if specific_event_type is not None:
            await self.trace_store.append_event(run_context, specific_event_type, payload)
        await self.trace_store.append_event(
            run_context,
            RolloutEventType.MODEL_CALL_FAILED,
            payload,
        )


def _chunk_text(text: str, chunk_size: int = 16) -> list[str]:
    if not text:
        return []
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
