from __future__ import annotations

from collections.abc import Iterable

from superbiz_agent.memory.ports import CoreContentWriteResult
from superbiz_agent.memory.schemas import CoreMemoryBlock, LongTermMemory
from superbiz_agent.memory.store import ExactMemoryWriteResult, InMemoryMemoryStore


class InMemoryMemoryRepository:
    """Thin async adapter over the process-local M-P0 store."""

    def __init__(self, store: InMemoryMemoryStore | None = None) -> None:
        self.store = store or InMemoryMemoryStore()

    async def ensure_ready(self) -> None:
        return None

    async def ensure_default_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[CoreMemoryBlock]:
        return self.store.ensure_default_core_blocks(tenant_id, user_id, agent_id)

    async def list_active_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[CoreMemoryBlock]:
        return self.store.load_core_blocks(tenant_id, user_id, agent_id)

    async def list_core_blocks_for_scope(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[CoreMemoryBlock]:
        return self.store.list_core_blocks_for_scope(
            tenant_id,
            user_id,
            agent_id,
            statuses=statuses,
        )

    async def cas_replace_core_content(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        block_key: str,
        content: str,
        *,
        expected_version: int,
    ) -> CoreContentWriteResult:
        status, block = self.store.cas_replace_core_content(
            tenant_id,
            user_id,
            agent_id,
            block_key,
            content,
            expected_version=expected_version,
        )
        return CoreContentWriteResult(status=status, block=block)

    async def write_archival_exact(
        self,
        memory: LongTermMemory,
    ) -> ExactMemoryWriteResult:
        return self.store.write_archival_exact(memory)

    async def list_active_memories(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        types: list[str] | None = None,
        scope_service: str | None = None,
        scope_env: str | None = None,
        tags: list[str] | None = None,
    ) -> list[LongTermMemory]:
        return self.store.active_memories(
            tenant_id,
            user_id,
            agent_id,
            types=types,
            scope_service=scope_service,
            scope_env=scope_env,
            tags=tags,
        )

    async def list_memories_for_scope(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[LongTermMemory]:
        return self.store.list_memories_for_scope(
            tenant_id,
            user_id,
            agent_id,
            statuses=statuses,
        )

    async def mark_returned(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        memory_ids: list[str],
    ) -> None:
        self.store.mark_returned(tenant_id, user_id, agent_id, memory_ids)

    async def upsert_core_block(self, block: CoreMemoryBlock) -> CoreMemoryBlock:
        return self.store.upsert_core_block(block)

    async def insert_memory(self, memory: LongTermMemory) -> LongTermMemory:
        return self.store.insert_memory(memory)
