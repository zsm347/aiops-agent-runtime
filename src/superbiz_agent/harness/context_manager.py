from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from superbiz_agent.harness.compaction import DeterministicHistoryCompactor
from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.context_budget import ContextBudget
from superbiz_agent.harness.events import RolloutEvent
from superbiz_agent.harness.token_estimator import ApproxTokenEstimator, TokenEstimator
from superbiz_agent.harness.tool_result_reducer import ToolResultReducer
from superbiz_agent.memory.schemas import MemoryContext
from superbiz_agent.model_gateway.base import ModelMessage
from superbiz_agent.prompts.registry import PromptRegistry


class MemoryContextProvider(Protocol):
    def build_context(self, tenant_id: str, user_id: str, agent_id: str) -> MemoryContext:
        ...


@dataclass(frozen=True)
class ContextComponentUsage:
    name: str
    required: bool
    priority: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    included: bool
    action: str
    dropped_reason: str | None = None

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            "priority": self.priority,
            "estimatedTokensBefore": self.estimated_tokens_before,
            "estimatedTokensAfter": self.estimated_tokens_after,
            "included": self.included,
            "action": self.action,
            "droppedReason": self.dropped_reason,
        }


@dataclass(frozen=True)
class PreparedContext:
    system_prompt: str
    prompt_version: str
    memory_messages: list[ModelMessage]
    summary_messages: list[ModelMessage]
    history_messages: list[ModelMessage]
    current_user_message: ModelMessage
    component_usages: list[ContextComponentUsage]
    estimated_input_tokens: int
    estimated_chars: int
    budget: ContextBudget
    compaction_triggered: bool
    tool_results_reduced: bool
    overflow: bool
    history_trim_event_payload: dict[str, Any] | None = None
    tool_result_reduction_payload: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    original_history_item_count: int = 0
    has_core_memory: bool = False
    has_memory_index: bool = False
    has_memory_metadata: bool = False
    core_memory_block_count: int = 0
    memory_index_topic_count: int = 0

    @property
    def visible_history_messages(self) -> list[ModelMessage]:
        return [*self.summary_messages, *self.history_messages]


class ContextManager:
    def __init__(
        self,
        *,
        prompt_registry: PromptRegistry,
        budget: ContextBudget,
        tool_result_reducer: ToolResultReducer,
        memory_context_provider: MemoryContextProvider | None = None,
        estimator: TokenEstimator | None = None,
        history_compactor: DeterministicHistoryCompactor | None = None,
    ) -> None:
        self.prompt_registry = prompt_registry
        self.memory_context_provider = memory_context_provider
        self.budget = budget
        self.estimator = estimator or ApproxTokenEstimator()
        self.tool_result_reducer = tool_result_reducer
        self.history_compactor = history_compactor or DeterministicHistoryCompactor()

    def prepare(
        self,
        *,
        run_context: RunContext,
        prompt_version: str,
        active_history: list[ModelMessage],
        current_user_message: str,
        history_events: list[RolloutEvent],
    ) -> PreparedContext:
        system_prompt = self.prompt_registry.load(prompt_version)
        system_message = ModelMessage(role="system", content=system_prompt)
        memory_context = self._build_memory_context(run_context.request_context)
        memory_messages = self._memory_messages(memory_context)
        current_message = ModelMessage(role="user", content=current_user_message)

        summary_messages, reducible_history = _extract_summary_messages(active_history)
        tool_reduction = self.tool_result_reducer.reduce(
            reducible_history,
            run_context=run_context,
        )
        candidate_summary = summary_messages
        candidate_history = tool_reduction.messages
        candidate_messages = [
            system_message,
            *memory_messages,
            *candidate_summary,
            *candidate_history,
            current_message,
        ]
        estimated_after_tool = self.estimator.estimate_messages(candidate_messages)

        compaction_triggered = False
        history_trim_payload = None
        warnings = list(tool_reduction.warnings)
        compacted_source_count = 0
        compaction_before = self.estimator.estimate_messages(
            [*candidate_summary, *candidate_history]
        )
        compaction_after = compaction_before

        if self.budget.should_compact(estimated_after_tool):
            compaction_input = [*candidate_summary, *candidate_history]
            compaction = self.history_compactor.compact(
                compaction_input,
                budget=self.budget,
                estimator=self.estimator,
            )
            if compaction.compacted and compaction.summary_message is not None:
                candidate_summary = [compaction.summary_message]
                candidate_history = compaction.recent_messages
                compaction_triggered = True
                compacted_source_count = len(compaction.source_messages)
                compaction_before = compaction.estimated_tokens_before
                compaction_after = compaction.estimated_tokens_after
                history_trim_payload = self._history_trim_payload(
                    history_events=history_events,
                    summary=compaction.summary_message.content,
                    source_message_count=compacted_source_count,
                    recent_messages_kept=len(candidate_history),
                    estimated_tokens_before=estimated_after_tool,
                    estimated_tokens_after=self.estimator.estimate_messages(
                        [
                            system_message,
                            *memory_messages,
                            *candidate_summary,
                            *candidate_history,
                            current_message,
                        ]
                    ),
                    tool_results_reduced=tool_reduction.changed,
                )
            else:
                warnings.append("context budget exceeded but no old history window was compactable")

        final_messages = [
            system_message,
            *memory_messages,
            *candidate_summary,
            *candidate_history,
            current_message,
        ]
        estimated_final = self.estimator.estimate_messages(final_messages)
        component_usages = self._component_usages(
            system_message=system_message,
            memory_context=memory_context,
            memory_messages=memory_messages,
            active_history=active_history,
            summary_messages=candidate_summary,
            history_messages=candidate_history,
            current_message=current_message,
            estimated_active_history_after_tool=tool_reduction.after_tokens,
            tool_reduction_before=tool_reduction.before_tokens,
            tool_reduction_after=tool_reduction.after_tokens,
            compaction_triggered=compaction_triggered,
            compaction_before=compaction_before,
        )
        overflow = estimated_final > self.budget.effective_input_budget
        if overflow:
            warnings.append("prepared context remains over effective input budget")

        return PreparedContext(
            system_prompt=system_prompt,
            prompt_version=prompt_version,
            memory_messages=memory_messages,
            summary_messages=candidate_summary,
            history_messages=candidate_history,
            current_user_message=current_message,
            component_usages=component_usages,
            estimated_input_tokens=estimated_final,
            estimated_chars=sum(len(message.content) for message in final_messages),
            budget=self.budget,
            compaction_triggered=compaction_triggered,
            tool_results_reduced=tool_reduction.changed,
            overflow=overflow,
            history_trim_event_payload=history_trim_payload,
            tool_result_reduction_payload=tool_reduction.to_trace_payload(),
            warnings=warnings,
            original_history_item_count=len(active_history),
            has_core_memory=bool(memory_context and memory_context.has_core_memory),
            has_memory_index=bool(memory_context and memory_context.has_memory_index),
            has_memory_metadata=bool(memory_context and memory_context.has_memory_index),
            core_memory_block_count=memory_context.core_block_count
            if memory_context
            else 0,
            memory_index_topic_count=memory_context.memory_index_topic_count
            if memory_context
            else 0,
        )

    def _build_memory_context(self, request_context: AgentRequestContext) -> MemoryContext | None:
        if self.memory_context_provider is None:
            return None
        return self.memory_context_provider.build_context(
            request_context.tenant_id or "",
            request_context.user_id or "",
            request_context.agent_id or "",
        )

    @staticmethod
    def _memory_messages(memory_context: MemoryContext | None) -> list[ModelMessage]:
        if memory_context is None:
            return []
        messages: list[ModelMessage] = []
        if memory_context.core_memory_xml.strip():
            messages.append(ModelMessage(role="system", content=memory_context.core_memory_xml))
        if memory_context.memory_index_xml.strip():
            messages.append(ModelMessage(role="system", content=memory_context.memory_index_xml))
        return messages

    def _history_trim_payload(
        self,
        *,
        history_events: list[RolloutEvent],
        summary: str,
        source_message_count: int,
        recent_messages_kept: int,
        estimated_tokens_before: int,
        estimated_tokens_after: int,
        tool_results_reduced: bool,
    ) -> dict[str, Any]:
        replayable_sequences = [
            event.sequence
            for event in history_events
            if event.sequence > 0 and _is_replayable_or_summary_event(event)
        ]
        from_sequence = min(replayable_sequences) if replayable_sequences else 0
        to_sequence = max(replayable_sequences) if replayable_sequences else 0
        reduction_ratio = 0.0
        if estimated_tokens_before > 0:
            reduction_ratio = max(
                0.0,
                1.0 - (estimated_tokens_after / estimated_tokens_before),
            )
        return {
            "summary": summary,
            "compactionMode": "recent_turns",
            "compactionPromptVersion": self.budget.compaction_prompt_version,
            "triggerReason": "estimated_input_tokens_exceeded",
            "fromSequence": from_sequence,
            "toSequence": to_sequence,
            "sourceMessageCount": source_message_count,
            "recentMessagesKept": recent_messages_kept,
            "estimatedTokensBefore": estimated_tokens_before,
            "estimatedTokensAfter": estimated_tokens_after,
            "tokenReductionRatio": reduction_ratio,
            "toolResultsReduced": tool_results_reduced,
            "status": "success",
        }

    def _component_usages(
        self,
        *,
        system_message: ModelMessage,
        memory_context: MemoryContext | None,
        memory_messages: list[ModelMessage],
        active_history: list[ModelMessage],
        summary_messages: list[ModelMessage],
        history_messages: list[ModelMessage],
        current_message: ModelMessage,
        estimated_active_history_after_tool: int,
        tool_reduction_before: int,
        tool_reduction_after: int,
        compaction_triggered: bool,
        compaction_before: int,
    ) -> list[ContextComponentUsage]:
        usages = [
            ContextComponentUsage(
                name="system_prompt",
                required=True,
                priority=100,
                estimated_tokens_before=self.estimator.estimate_messages([system_message]),
                estimated_tokens_after=self.estimator.estimate_messages([system_message]),
                included=True,
                action="included",
            )
        ]
        if memory_context is not None:
            core_messages = [
                message
                for message in memory_messages
                if message.content.startswith("<core_memory>")
            ]
            index_messages = [
                message
                for message in memory_messages
                if message.content.startswith("<memory_metadata>")
            ]
            usages.append(
                ContextComponentUsage(
                    name="core_memory",
                    required=True,
                    priority=90,
                    estimated_tokens_before=self.estimator.estimate_messages(core_messages),
                    estimated_tokens_after=self.estimator.estimate_messages(core_messages),
                    included=bool(core_messages),
                    action="included" if core_messages else "empty",
                )
            )
            usages.append(
                ContextComponentUsage(
                    name="memory_index",
                    required=False,
                    priority=80,
                    estimated_tokens_before=self.estimator.estimate_messages(index_messages),
                    estimated_tokens_after=self.estimator.estimate_messages(index_messages),
                    included=bool(index_messages),
                    action="included" if index_messages else "empty",
                )
            )
        usages.append(
            ContextComponentUsage(
                name="active_history",
                required=False,
                priority=50,
                estimated_tokens_before=self.estimator.estimate_messages(active_history),
                estimated_tokens_after=self.estimator.estimate_messages(
                    [*summary_messages, *history_messages]
                ),
                included=bool(summary_messages or history_messages),
                action="compacted" if compaction_triggered else "included",
            )
        )
        usages.append(
            ContextComponentUsage(
                name="tool_results",
                required=False,
                priority=45,
                estimated_tokens_before=tool_reduction_before,
                estimated_tokens_after=tool_reduction_after,
                included=tool_reduction_after > 0,
                action="reduced" if tool_reduction_before != tool_reduction_after else "included",
            )
        )
        usages.append(
            ContextComponentUsage(
                name="compacted_summary",
                required=False,
                priority=60,
                estimated_tokens_before=compaction_before,
                estimated_tokens_after=self.estimator.estimate_messages(summary_messages),
                included=bool(summary_messages),
                action="generated"
                if compaction_triggered
                else "included"
                if summary_messages
                else "empty",
            )
        )
        usages.append(
            ContextComponentUsage(
                name="recent_history",
                required=False,
                priority=70,
                estimated_tokens_before=estimated_active_history_after_tool,
                estimated_tokens_after=self.estimator.estimate_messages(history_messages),
                included=bool(history_messages),
                action="kept",
            )
        )
        usages.append(
            ContextComponentUsage(
                name="current_user_message",
                required=True,
                priority=100,
                estimated_tokens_before=self.estimator.estimate_messages([current_message]),
                estimated_tokens_after=self.estimator.estimate_messages([current_message]),
                included=True,
                action="included",
            )
        )
        return usages


def _extract_summary_messages(
    active_history: list[ModelMessage],
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    summary_messages: list[ModelMessage] = []
    index = 0
    while index < len(active_history):
        message = active_history[index]
        if message.role == "system" and (
            message.metadata.get("context_summary") is True
            or message.content.lstrip().startswith("<conversation_summary>")
        ):
            summary_messages.append(message)
            index += 1
            continue
        break
    return summary_messages, active_history[index:]


def _is_replayable_or_summary_event(event: RolloutEvent) -> bool:
    return event.event_type.value in {
        "USER_MESSAGE_APPENDED",
        "ASSISTANT_MESSAGE_APPENDED",
        "TOOL_CALL_STARTED",
        "TOOL_CALL_COMPLETED",
        "TOOL_CALL_FAILED",
        "TOOL_CALL_BLOCKED",
        "HISTORY_TRIMMED",
    }
