from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.memory.schemas import MemoryContext
from superbiz_agent.model_gateway.base import ModelMessage
from superbiz_agent.prompts.registry import PromptRegistry


class MemoryContextProvider(Protocol):
    def build_context(self, tenant_id: str, user_id: str, agent_id: str) -> MemoryContext:
        ...


@dataclass(frozen=True)
class AssembledContext:
    messages: list[ModelMessage]
    prompt_version: str
    estimated_chars: int
    history_item_count: int
    has_core_memory: bool = False
    has_memory_index: bool = False
    has_memory_metadata: bool = False
    core_memory_block_count: int = 0
    memory_index_topic_count: int = 0
    estimated_input_tokens: int = 0
    budget_payload: dict = field(default_factory=dict)
    compaction_triggered: bool = False
    tool_results_reduced: bool = False
    overflow: bool = False
    component_usages: list[dict] = field(default_factory=list)
    tool_result_reduction: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def trace_payload(self) -> dict:
        return {
            "promptVersion": self.prompt_version,
            "historyItemCount": self.history_item_count,
            "messageCount": len(self.messages),
            "estimatedChars": self.estimated_chars,
            "hasCoreMemory": self.has_core_memory,
            "hasMemoryIndex": self.has_memory_index,
            "hasMemoryMetadata": self.has_memory_metadata,
            "coreMemoryBlockCount": self.core_memory_block_count,
            "memoryIndexTopicCount": self.memory_index_topic_count,
            "estimatedInputTokens": self.estimated_input_tokens,
            "tokenBudget": self.budget_payload,
            "compactionTriggered": self.compaction_triggered,
            "toolResultsReduced": self.tool_results_reduced,
            "overflow": self.overflow,
            "componentUsage": self.component_usages,
            "componentUsages": self.component_usages,
            "toolResultReduction": self.tool_result_reduction,
            "warnings": self.warnings,
        }


class ContextAssembler:
    def __init__(
        self,
        prompt_registry: PromptRegistry,
        memory_context_provider: MemoryContextProvider | None = None,
    ) -> None:
        self.prompt_registry = prompt_registry
        self.memory_context_provider = memory_context_provider

    def assemble(
        self,
        *,
        prompt_version: str | None = None,
        active_history: list[ModelMessage] | None = None,
        current_user_message: str | None = None,
        request_context: AgentRequestContext | None = None,
        prepared_context: object | None = None,
    ) -> AssembledContext:
        if prepared_context is not None:
            return self.assemble_prepared(prepared_context)
        if prompt_version is None or current_user_message is None:
            raise ValueError("prompt_version and current_user_message are required")
        active_history = active_history or []
        system_prompt = self.prompt_registry.load(prompt_version)
        messages = [ModelMessage(role="system", content=system_prompt)]
        memory_context = None
        if self.memory_context_provider is not None and request_context is not None:
            memory_context = self.memory_context_provider.build_context(
                request_context.tenant_id or "",
                request_context.user_id or "",
                request_context.agent_id or "",
            )
            if memory_context.core_memory_xml.strip():
                messages.append(
                    ModelMessage(role="system", content=memory_context.core_memory_xml)
                )
            if memory_context.memory_index_xml.strip():
                messages.append(
                    ModelMessage(role="system", content=memory_context.memory_index_xml)
                )
        messages.extend(active_history)
        messages.append(ModelMessage(role="user", content=current_user_message))
        return AssembledContext(
            messages=messages,
            prompt_version=prompt_version,
            estimated_chars=sum(len(message.content) for message in messages),
            history_item_count=len(active_history),
            has_core_memory=bool(memory_context and memory_context.has_core_memory),
            has_memory_index=bool(memory_context and memory_context.has_memory_index),
            has_memory_metadata=bool(memory_context and memory_context.has_memory_index),
            core_memory_block_count=memory_context.core_block_count if memory_context else 0,
            memory_index_topic_count=memory_context.memory_index_topic_count if memory_context else 0,
        )

    def assemble_prepared(self, prepared_context: object) -> AssembledContext:
        messages = [
            ModelMessage(role="system", content=prepared_context.system_prompt),
            *prepared_context.memory_messages,
            *prepared_context.summary_messages,
            *prepared_context.history_messages,
            prepared_context.current_user_message,
        ]
        return AssembledContext(
            messages=messages,
            prompt_version=prepared_context.prompt_version,
            estimated_chars=prepared_context.estimated_chars,
            history_item_count=prepared_context.original_history_item_count,
            has_core_memory=prepared_context.has_core_memory,
            has_memory_index=prepared_context.has_memory_index,
            has_memory_metadata=getattr(
                prepared_context,
                "has_memory_metadata",
                prepared_context.has_memory_index,
            ),
            core_memory_block_count=prepared_context.core_memory_block_count,
            memory_index_topic_count=prepared_context.memory_index_topic_count,
            estimated_input_tokens=prepared_context.estimated_input_tokens,
            budget_payload=prepared_context.budget.to_trace_payload(),
            compaction_triggered=prepared_context.compaction_triggered,
            tool_results_reduced=prepared_context.tool_results_reduced,
            overflow=prepared_context.overflow,
            component_usages=[
                usage.to_trace_payload() for usage in prepared_context.component_usages
            ],
            tool_result_reduction=prepared_context.tool_result_reduction_payload,
            warnings=prepared_context.warnings,
        )
