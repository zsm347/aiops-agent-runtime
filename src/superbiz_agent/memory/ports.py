from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from superbiz_agent.memory.schemas import CoreMemoryBlock, LongTermMemory
from superbiz_agent.memory.store import ExactMemoryWriteResult


@dataclass(frozen=True)
class CoreContentWriteResult:
    status: Literal["updated", "unchanged", "conflict", "inactive", "read_only"]
    block: CoreMemoryBlock | None


class MemoryInspectionRepository(Protocol):
    async def list_core_blocks_for_scope(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[CoreMemoryBlock]: ...

    async def list_memories_for_scope(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[LongTermMemory]: ...


class MemoryRepository(MemoryInspectionRepository, Protocol):
    async def ensure_ready(self) -> None: ...

    async def ensure_default_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[CoreMemoryBlock]: ...

    async def list_active_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[CoreMemoryBlock]: ...

    async def cas_replace_core_content(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        block_key: str,
        content: str,
        *,
        expected_version: int,
    ) -> CoreContentWriteResult: ...

    async def write_archival_exact(
        self,
        memory: LongTermMemory,
    ) -> ExactMemoryWriteResult: ...

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
    ) -> list[LongTermMemory]: ...

    async def mark_returned(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        memory_ids: list[str],
    ) -> None: ...


class MemoryFixtureAdmin(Protocol):
    async def ensure_default_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[CoreMemoryBlock]: ...

    async def upsert_core_block(self, block: CoreMemoryBlock) -> CoreMemoryBlock: ...

    async def insert_memory(self, memory: LongTermMemory) -> LongTermMemory: ...

