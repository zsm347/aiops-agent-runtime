from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncEngine

from superbiz_agent.config import Settings
from superbiz_agent.harness.stores import RolloutEventStore
from superbiz_agent.memory.archival import ArchivalMemoryService
from superbiz_agent.memory.adapters.in_memory import InMemoryMemoryRepository
from superbiz_agent.memory.core import CoreMemoryService
from superbiz_agent.memory.embedding import DeterministicEmbeddingService
from superbiz_agent.memory.index import MemoryContextProvider, MemoryIndexService
from superbiz_agent.memory.policy import MemoryWritePolicy
from superbiz_agent.memory.ports import (
    MemoryFixtureAdmin,
    MemoryInspectionRepository,
    MemoryRepository,
)
from superbiz_agent.memory.run_snapshots import CoreVersionSnapshotRegistry
from superbiz_agent.memory.search import MemorySearchService
from superbiz_agent.memory.store import InMemoryMemoryStore
from superbiz_agent.memory.tools import build_memory_tools
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository
from superbiz_agent.tools.registry import ToolDefinition


@dataclass
class MemoryRuntimeComponents:
    """Long-term-memory wiring shared by the production Harness and eval fixtures."""

    backend: str
    repository: MemoryRepository
    inspection_repository: MemoryInspectionRepository
    fixture_admin: MemoryFixtureAdmin | None
    core_version_snapshots: CoreVersionSnapshotRegistry
    policy: MemoryWritePolicy
    embedding_service: DeterministicEmbeddingService
    core_service: CoreMemoryService
    archival_service: ArchivalMemoryService
    search_service: MemorySearchService
    index_service: MemoryIndexService
    context_provider: MemoryContextProvider
    tools: tuple[ToolDefinition, ...]
    trace_store: RolloutEventStore
    engine: AsyncEngine | None = None
    _ready_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _ready_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    async def ensure_ready(self) -> None:
        """Fail closed if the persistence backend does not satisfy the contract.

        For the ``postgres`` backend this runs the named-schema capability probe.
        For the ``memory`` backend the repository is always ready.
        """
        async with self._ready_lock:
            if self._ready_task is None:
                self._ready_task = asyncio.create_task(self.repository.ensure_ready())
            task = self._ready_task
        await asyncio.shield(task)

    async def aclose(self) -> None:
        """Dispose a PostgreSQL engine; a no-op for the in-memory backend."""
        async with self._close_lock:
            if self._closed:
                return
            task = self._close_task
            if task is None or task.done():
                task = asyncio.create_task(self._dispose_engine())
                self._close_task = task
        await asyncio.shield(task)

    async def _dispose_engine(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()
        async with self._close_lock:
            self._closed = True


def build_memory_runtime(
    settings: Settings,
    *,
    trace_store: RolloutEventStore,
) -> MemoryRuntimeComponents:
    """Create one isolated long-term-memory runtime for the configured backend."""

    backend = settings.memory_store_backend
    if backend == "postgres":
        from superbiz_agent.persistence.database import (
            create_engine,
            create_sessionmaker,
        )

        engine = create_engine(settings.database_url)
        sessionmaker = create_sessionmaker(engine)
        repository: MemoryRepository = PostgresMemoryRepository(sessionmaker)
        fixture_admin: MemoryFixtureAdmin | None = None
    elif backend == "memory":
        store = InMemoryMemoryStore()
        adapter = InMemoryMemoryRepository(store)
        repository = adapter
        fixture_admin = adapter
        engine = None
    else:  # pragma: no cover - guarded by Settings validation
        raise ValueError(f"unsupported memory_store_backend: {backend!r}")

    policy = MemoryWritePolicy()
    embedding_service = DeterministicEmbeddingService(
        dimension=settings.memory_embedding_dimension
    )
    snapshots = CoreVersionSnapshotRegistry()
    core_service = CoreMemoryService(repository, policy, snapshots)
    search_service = MemorySearchService(
        repository,
        embedding_service,
        top_k=settings.memory_search_top_k,
        min_similarity=settings.memory_search_min_similarity,
    )
    archival_service = ArchivalMemoryService(
        repository,
        policy,
        embedding_service,
    )
    index_service = MemoryIndexService(repository, search_service)
    context_provider = MemoryContextProvider(core_service, index_service, snapshots)
    tools = tuple(
        build_memory_tools(
            core_service=core_service,
            archival_service=archival_service,
            search_service=search_service,
            trace_store=trace_store,
        )
    )
    return MemoryRuntimeComponents(
        backend=backend,
        repository=repository,
        inspection_repository=repository,
        fixture_admin=fixture_admin,
        core_version_snapshots=snapshots,
        policy=policy,
        embedding_service=embedding_service,
        core_service=core_service,
        archival_service=archival_service,
        search_service=search_service,
        index_service=index_service,
        context_provider=context_provider,
        tools=tools,
        trace_store=trace_store,
        engine=engine,
    )
