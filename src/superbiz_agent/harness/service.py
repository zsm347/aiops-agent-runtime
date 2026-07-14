from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional

from superbiz_agent.api.schemas import SseMessage
from superbiz_agent.config import Settings
from superbiz_agent.harness.content_compression import build_content_compression_backend
from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.context_assembler import ContextAssembler
from superbiz_agent.harness.context_budget import context_budget_from_settings
from superbiz_agent.harness.context_manager import ContextManager
from superbiz_agent.harness.graph import SkeletonAgentGraph
from superbiz_agent.harness.history import count_message_pairs
from superbiz_agent.harness.locks import ConversationLockManager
from superbiz_agent.harness.runtime import ConversationRuntime
from superbiz_agent.harness.stores import PostgresRolloutEventStore, RolloutEventStore
from superbiz_agent.harness.token_estimator import ApproxTokenEstimator
from superbiz_agent.harness.tool_result_reducer import ToolResultReducer
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore
from superbiz_agent.memory.runtime import MemoryRuntimeComponents, build_memory_runtime
from superbiz_agent.model_gateway.base import ModelGateway
from superbiz_agent.model_gateway.factory import build_model_gateway
from superbiz_agent.prompts.registry import PromptRegistry
from superbiz_agent.rag.retrieval import (
    FixtureRagRetrievalService,
    RagRetrievalService,
    UnavailableRagRetrievalService,
)
from superbiz_agent.tools.builtin import build_builtin_tools
from superbiz_agent.tools.gateway import ToolGateway
from superbiz_agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class AgentChatResult:
    success: bool
    session_id: str
    run_id: Optional[str] = None
    answer: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    message_pair_count: int
    create_time: int = 0


class AgentHarnessService:
    """Skeleton P0 Harness service with deterministic model and tool loop."""

    def __init__(
        self,
        *,
        runtime: ConversationRuntime,
        graph: SkeletonAgentGraph,
        trace_store: RolloutEventStore,
        memory_runtime: MemoryRuntimeComponents | None = None,
        rag_retrieval_service: RagRetrievalService | None = None,
        lock_manager: ConversationLockManager | None = None,
    ) -> None:
        self.runtime = runtime
        self.graph = graph
        self.trace_store = trace_store
        self.memory_runtime = memory_runtime
        self.rag_retrieval_service = rag_retrieval_service
        self.lock_manager = lock_manager or ConversationLockManager()
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False

    @classmethod
    def build_default(
        cls,
        settings: Settings,
        *,
        model_gateway: ModelGateway | None = None,
        trace_store: RolloutEventStore | None = None,
        memory_runtime: MemoryRuntimeComponents | None = None,
        rag_retrieval_service: RagRetrievalService | None = None,
    ) -> "AgentHarnessService":
        tool_registry = ToolRegistry(schema_version=settings.tool_schema_version)
        prompt_registry = PromptRegistry(settings.resolve_prompt_dir())
        resolved_trace_store = trace_store or (
            memory_runtime.trace_store if memory_runtime is not None else cls._build_trace_store(settings)
        )
        memory_tools = []
        memory_context_provider = None
        if settings.memory_enabled:
            memory_runtime = memory_runtime or build_memory_runtime(
                settings,
                trace_store=resolved_trace_store,
            )
            if memory_runtime.trace_store is not resolved_trace_store:
                raise ValueError("memory_runtime and Harness must use the same trace_store")
            memory_context_provider = memory_runtime.context_provider
            memory_tools = list(memory_runtime.tools)
        elif memory_runtime is not None:
            raise ValueError("memory_runtime cannot be supplied when memory is disabled")
        context_assembler = ContextAssembler(prompt_registry, memory_context_provider)
        context_budget = context_budget_from_settings(settings)
        token_estimator = ApproxTokenEstimator()
        compression_backend = build_content_compression_backend(
            context_budget.content_compression_backend
        )
        tool_result_reducer = ToolResultReducer(
            budget=context_budget,
            estimator=token_estimator,
            compression_backend=compression_backend,
        )
        context_manager = ContextManager(
            prompt_registry=prompt_registry,
            memory_context_provider=memory_context_provider,
            budget=context_budget,
            estimator=token_estimator,
            tool_result_reducer=tool_result_reducer,
        )
        resolved_model_gateway = model_gateway or build_model_gateway(settings)

        resolved_rag_retrieval_service = rag_retrieval_service or cls._build_rag_retrieval_service(
            settings
        )
        for definition in build_builtin_tools(
            memory_tools,
            rag_retrieval_service=resolved_rag_retrieval_service,
        ):
            tool_registry.register(definition)
        tool_registry.seal_schema()
        tool_gateway = ToolGateway(tool_registry, resolved_trace_store, settings=settings)

        runtime = ConversationRuntime(
            context_assembler=context_assembler,
            trace_store=resolved_trace_store,
            prompt_version=settings.prompt_version,
            tool_schema_version=tool_registry.schema_version or "",
            model_provider=settings.model_provider,
            context_manager=context_manager,
            core_version_snapshots=(
                memory_runtime.core_version_snapshots if memory_runtime is not None else None
            ),
        )
        graph = SkeletonAgentGraph(
            model_gateway=resolved_model_gateway,
            tool_gateway=tool_gateway,
            trace_store=resolved_trace_store,
            tool_result_reducer=tool_result_reducer,
            max_tool_rounds=settings.agent_max_tool_rounds,
            max_tool_calls_per_run=settings.agent_max_tool_calls_per_run,
        )
        return cls(
            runtime=runtime,
            graph=graph,
            trace_store=resolved_trace_store,
            memory_runtime=memory_runtime,
            rag_retrieval_service=resolved_rag_retrieval_service,
            lock_manager=ConversationLockManager(),
        )

    @staticmethod
    def _build_trace_store(settings: Settings) -> RolloutEventStore:
        if settings.rollout_store_backend == "postgres":
            from superbiz_agent.persistence.database import create_engine, create_sessionmaker
            from superbiz_agent.persistence.repositories.rollout_events import RolloutEventRepository

            engine = create_engine(settings.database_url)
            sessionmaker = create_sessionmaker(engine)
            repository = RolloutEventRepository(sessionmaker)
            return PostgresRolloutEventStore(repository)
        return InMemoryRolloutEventStore()

    @staticmethod
    def _build_rag_retrieval_service(settings: Settings) -> RagRetrievalService:
        if settings.rag_fixture_mode:
            return FixtureRagRetrievalService()
        if settings.rag_enabled:
            from superbiz_agent.rag.runtime import build_real_rag_retrieval_service

            return build_real_rag_retrieval_service(settings)
        return UnavailableRagRetrievalService("Internal document retrieval is disabled.")

    @classmethod
    def placeholder(cls) -> "AgentHarnessService":
        from superbiz_agent.config import get_settings

        return cls.build_default(get_settings())

    async def chat(self, context: AgentRequestContext, question: str) -> AgentChatResult:
        if not question.strip():
            return AgentChatResult(
                success=False,
                session_id=context.session_id or "",
                error_message="问题内容不能为空",
            )

        async with self.lock_manager.acquire(context):
            run_context = None
            try:
                run_context = await self.runtime.start_run(context, question)
                await self.runtime.append_user_message(run_context, question)
                assembled = await self.runtime.assemble_context(run_context, question)
                answer = await self.graph.run(run_context, assembled.messages)
                await self.runtime.append_assistant_message(run_context, answer)
                await self.runtime.complete_run(run_context, answer)
                return AgentChatResult(
                    success=True,
                    session_id=context.session_id or "",
                    run_id=run_context.run_id,
                    answer=answer,
                )
            except asyncio.CancelledError:
                if run_context is not None:
                    await self._best_effort_fail_run(run_context, "request cancelled")
                raise
            except Exception as exc:
                if run_context is not None:
                    await self._best_effort_fail_run(run_context, str(exc))
                return AgentChatResult(
                    success=False,
                    session_id=context.session_id or "",
                    run_id=run_context.run_id if run_context is not None else None,
                    error_message=str(exc),
                )
            finally:
                if run_context is not None:
                    self._cleanup_run(run_context.run_id)

    async def chat_stream(
        self,
        context: AgentRequestContext,
        question: str,
    ) -> AsyncIterator[SseMessage]:
        if not question.strip():
            yield SseMessage(type="error", data="问题内容不能为空")
            yield SseMessage(type="done", data=None)
            return

        async with self.lock_manager.acquire(context):
            run_context = None
            answer = ""
            try:
                run_context = await self.runtime.start_run(context, question)
                await self.runtime.append_user_message(run_context, question)
                assembled = await self.runtime.assemble_context(run_context, question)
                async for event in self.graph.run_stream(run_context, assembled.messages):
                    if event.type == "content":
                        if isinstance(event.data, str):
                            answer += event.data
                        yield SseMessage(type="content", data=event.data)
                    elif event.type == "final":
                        answer = _extract_final_answer(event.data, answer)
                    else:
                        yield SseMessage(type=event.type, data=event.data)

                await self.runtime.append_assistant_message(run_context, answer)
                await self.runtime.complete_run(run_context, answer)
                yield SseMessage(type="final", data={"data": answer})
                yield SseMessage(type="done", data=None)
            except asyncio.CancelledError:
                if run_context is not None:
                    await self._best_effort_fail_run(run_context, "request cancelled")
                raise
            except Exception as exc:
                if run_context is not None:
                    await self._best_effort_fail_run(run_context, str(exc))
                yield SseMessage(type="error", data=str(exc))
                yield SseMessage(type="done", data=None)
            finally:
                if run_context is not None:
                    self._cleanup_run(run_context.run_id)

    async def _best_effort_fail_run(
        self,
        run_context: RunContext,
        error_message: str,
    ) -> None:
        try:
            await self.runtime.fail_run(run_context, error_message)
        except Exception:
            self.runtime.cleanup_run(run_context.run_id)

    def _cleanup_run(self, run_id: str) -> None:
        self.runtime.cleanup_run(run_id)
        graph_cleanup = getattr(self.graph, "cleanup_run", None)
        if callable(graph_cleanup):
            graph_cleanup(run_id)

    async def clear(self, context: AgentRequestContext) -> None:
        await self.trace_store.clear_session(context)

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            task = self._close_task
            if task is None or task.done():
                task = asyncio.create_task(self._close_resources())
                self._close_task = task
        await asyncio.shield(task)

    async def _close_resources(self) -> None:
        close = getattr(self.rag_retrieval_service, "aclose", None)
        if callable(close):
            try:
                await close()
            except (Exception, asyncio.CancelledError):
                raise RuntimeError("Failed to close Harness resources: rag_runtime.") from None
        async with self._close_lock:
            self._closed = True

    async def session_info(self, context: AgentRequestContext) -> SessionInfo:
        events = await self.trace_store.list_by_session(context)
        return SessionInfo(
            session_id=context.session_id or "",
            message_pair_count=count_message_pairs(events),
            create_time=0,
        )


def _extract_final_answer(data: object, fallback: str) -> str:
    if isinstance(data, dict):
        answer = data.get("data")
        if isinstance(answer, str):
            return answer
    if isinstance(data, str):
        return data
    return fallback
