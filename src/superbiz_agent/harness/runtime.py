from __future__ import annotations

from uuid import uuid4

from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.context_assembler import AssembledContext, ContextAssembler
from superbiz_agent.harness.context_manager import ContextManager
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.history import recover_active_history
from superbiz_agent.harness.stores import RolloutEventStore
from superbiz_agent.memory.run_snapshots import CoreVersionSnapshotRegistry
from superbiz_agent.model_gateway.base import ModelMessage


class ConversationRuntime:
    def __init__(
        self,
        *,
        context_assembler: ContextAssembler,
        trace_store: RolloutEventStore,
        prompt_version: str,
        tool_schema_version: str,
        model_provider: str,
        context_manager: ContextManager | None = None,
        core_version_snapshots: CoreVersionSnapshotRegistry | None = None,
    ) -> None:
        self.context_assembler = context_assembler
        self.context_manager = context_manager
        self.trace_store = trace_store
        self.prompt_version = prompt_version
        self.tool_schema_version = tool_schema_version
        self.model_provider = model_provider
        self.core_version_snapshots = core_version_snapshots
        self._active_history_by_run: dict[str, list[ModelMessage]] = {}
        self._history_events_by_run = {}

    async def start_run(self, context: AgentRequestContext, question: str = "") -> RunContext:
        previous_events = await self.trace_store.list_by_session(context)
        active_history = recover_active_history(previous_events)
        run_context = RunContext(
            request_context=context,
            run_id=str(uuid4()),
            prompt_version=self.prompt_version,
            tool_schema_version=self.tool_schema_version,
            model_provider=self.model_provider,
        )
        self._active_history_by_run[run_context.run_id] = active_history
        self._history_events_by_run[run_context.run_id] = previous_events
        try:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.RUN_STARTED,
                {
                    "question": question,
                    "promptVersion": self.prompt_version,
                    "toolSchemaVersion": self.tool_schema_version,
                    "modelProvider": self.model_provider,
                    "authMode": context.auth_mode,
                    "status": "started",
                },
            )
            if previous_events:
                await self.trace_store.append_event(
                    run_context,
                    RolloutEventType.THREAD_RECOVERED,
                    {
                        "historyItemCount": len(active_history),
                        "eventCount": len(previous_events),
                        "status": "success",
                    },
                )
        except BaseException:
            self.cleanup_run(run_context.run_id)
            raise
        return run_context

    async def append_user_message(self, run_context: RunContext, question: str) -> None:
        await self.trace_store.append_event(
            run_context,
            RolloutEventType.USER_MESSAGE_APPENDED,
            {"content": question, "messageLength": len(question)},
            message_id=str(uuid4()),
        )

    async def assemble_context(
        self,
        run_context: RunContext,
        current_user_message: str,
    ) -> AssembledContext:
        assembled = await self.context_assembler.assemble(
            prepared_context=await self._prepare_context_if_enabled(
                run_context,
                current_user_message,
            )
            if self.context_manager is not None
            else None,
            prompt_version=run_context.prompt_version,
            active_history=self._active_history_by_run.get(run_context.run_id, []),
            current_user_message=current_user_message,
            run_context=run_context,
        )
        await self.trace_store.append_event(
            run_context,
            RolloutEventType.CONTEXT_ASSEMBLED,
            assembled.trace_payload,
        )
        if assembled.has_core_memory or assembled.has_memory_index:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.MEMORY_INJECTED,
                {
                    "hasCoreMemory": assembled.has_core_memory,
                    "hasMemoryIndex": assembled.has_memory_index,
                    "hasMemoryMetadata": assembled.has_memory_metadata,
                    "coreMemoryBlockCount": assembled.core_memory_block_count,
                    "memoryIndexTopicCount": assembled.memory_index_topic_count,
                },
            )
        return assembled

    async def _prepare_context_if_enabled(
        self,
        run_context: RunContext,
        current_user_message: str,
    ):
        if self.context_manager is None:
            return None
        prepared = await self.context_manager.prepare(
            run_context=run_context,
            prompt_version=run_context.prompt_version,
            active_history=self._active_history_by_run.get(run_context.run_id, []),
            current_user_message=current_user_message,
            history_events=self._history_events_by_run.get(run_context.run_id, []),
        )
        if prepared.history_trim_event_payload is not None:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.HISTORY_TRIMMED,
                {
                    **prepared.history_trim_event_payload,
                    "componentUsages": [
                        usage.to_trace_payload() for usage in prepared.component_usages
                    ],
                },
            )
        if prepared.overflow:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.MODEL_CALL_CONTEXT_OVERFLOW,
                {
                    "provider": run_context.model_provider,
                    "estimatedInputTokens": prepared.estimated_input_tokens,
                    "effectiveInputBudgetTokens": prepared.budget.effective_input_budget,
                    "status": "preflight_warning",
                },
            )
        self._active_history_by_run[run_context.run_id] = prepared.visible_history_messages
        return prepared

    async def append_assistant_message(self, run_context: RunContext, answer: str) -> None:
        self._active_history_by_run.setdefault(run_context.run_id, []).append(
            ModelMessage(role="assistant", content=answer)
        )
        await self.trace_store.append_event(
            run_context,
            RolloutEventType.ASSISTANT_MESSAGE_APPENDED,
            {"content": answer, "messageLength": len(answer)},
            message_id=str(uuid4()),
        )

    async def complete_run(self, run_context: RunContext, answer: str) -> None:
        try:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.RUN_COMPLETED,
                {"answer": answer, "status": "success"},
            )
        finally:
            self.cleanup_run(run_context.run_id)

    async def fail_run(self, run_context: RunContext, error_message: str) -> None:
        try:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.RUN_FAILED,
                {"errorMessage": error_message},
            )
        finally:
            self.cleanup_run(run_context.run_id)

    def cleanup_run(self, run_id: str) -> None:
        self._active_history_by_run.pop(run_id, None)
        self._history_events_by_run.pop(run_id, None)
        if self.core_version_snapshots is not None:
            self.core_version_snapshots.cleanup(run_id)
