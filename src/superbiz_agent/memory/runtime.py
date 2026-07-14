from __future__ import annotations

from dataclasses import dataclass

from superbiz_agent.config import Settings
from superbiz_agent.harness.stores import RolloutEventStore
from superbiz_agent.memory.archival import ArchivalMemoryService
from superbiz_agent.memory.core import CoreMemoryService
from superbiz_agent.memory.embedding import DeterministicEmbeddingService
from superbiz_agent.memory.index import MemoryContextProvider, MemoryIndexService
from superbiz_agent.memory.policy import MemoryWritePolicy
from superbiz_agent.memory.search import MemorySearchService
from superbiz_agent.memory.store import InMemoryMemoryStore
from superbiz_agent.memory.tools import build_memory_tools
from superbiz_agent.tools.registry import ToolDefinition


@dataclass(frozen=True)
class MemoryRuntimeComponents:
    """Long-term-memory wiring shared by the production Harness and eval fixtures."""

    store: InMemoryMemoryStore
    policy: MemoryWritePolicy
    embedding_service: DeterministicEmbeddingService
    core_service: CoreMemoryService
    archival_service: ArchivalMemoryService
    search_service: MemorySearchService
    index_service: MemoryIndexService
    context_provider: MemoryContextProvider
    tools: tuple[ToolDefinition, ...]
    trace_store: RolloutEventStore


def build_memory_runtime(
    settings: Settings,
    *,
    trace_store: RolloutEventStore,
) -> MemoryRuntimeComponents:
    """Create one isolated in-memory long-term-memory runtime."""

    store = InMemoryMemoryStore()
    policy = MemoryWritePolicy()
    embedding_service = DeterministicEmbeddingService(
        dimension=settings.memory_embedding_dimension
    )
    core_service = CoreMemoryService(store, policy)
    search_service = MemorySearchService(
        store,
        embedding_service,
        top_k=settings.memory_search_top_k,
        min_similarity=settings.memory_search_min_similarity,
    )
    archival_service = ArchivalMemoryService(
        store,
        policy,
        embedding_service,
    )
    index_service = MemoryIndexService(store, search_service)
    context_provider = MemoryContextProvider(core_service, index_service)
    tools = tuple(
        build_memory_tools(
            core_service=core_service,
            archival_service=archival_service,
            search_service=search_service,
            trace_store=trace_store,
        )
    )
    return MemoryRuntimeComponents(
        store=store,
        policy=policy,
        embedding_service=embedding_service,
        core_service=core_service,
        archival_service=archival_service,
        search_service=search_service,
        index_service=index_service,
        context_provider=context_provider,
        tools=tools,
        trace_store=trace_store,
    )
