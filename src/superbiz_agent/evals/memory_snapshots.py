from __future__ import annotations

from dataclasses import dataclass

from superbiz_agent.memory.store import InMemoryMemoryStore


@dataclass(frozen=True)
class MemoryIdentityScope:
    tenant_id: str
    user_id: str
    agent_id: str


@dataclass(frozen=True)
class CoreMemoryBlockSnapshot:
    block_key: str
    content: str
    version: int
    content_hash: str
    max_tokens: int
    status: str


@dataclass(frozen=True)
class ArchivalMemorySnapshot:
    id: str
    type: str
    topic: str
    content: str
    content_hash: str
    tags: tuple[str, ...]
    scope_service: str | None
    scope_env: str | None
    status: str
    usage_count: int


@dataclass(frozen=True)
class MemorySnapshot:
    identity_scope: MemoryIdentityScope
    core_blocks: tuple[CoreMemoryBlockSnapshot, ...]
    archival_memories: tuple[ArchivalMemorySnapshot, ...]


class MemorySnapshotProvider:
    """Capture immutable eval snapshots through read-only store APIs."""

    def __init__(self, store: InMemoryMemoryStore) -> None:
        self.store = store

    def capture(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> MemorySnapshot:
        identity = MemoryIdentityScope(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
        )
        core_blocks = tuple(
            CoreMemoryBlockSnapshot(
                block_key=block.block_key,
                content=block.content,
                version=block.version,
                content_hash=block.content_hash,
                max_tokens=block.max_tokens,
                status=block.status,
            )
            for block in self.store.list_core_blocks_for_scope(
                tenant_id,
                user_id,
                agent_id,
            )
        )
        archival_memories = tuple(
            ArchivalMemorySnapshot(
                id=memory.id,
                type=memory.type,
                topic=memory.topic,
                content=memory.content,
                content_hash=memory.content_hash,
                tags=tuple(memory.tags),
                scope_service=memory.scope_service,
                scope_env=memory.scope_env,
                status=memory.status,
                usage_count=memory.usage_count,
            )
            for memory in self.store.list_memories_for_scope(
                tenant_id,
                user_id,
                agent_id,
            )
        )
        return MemorySnapshot(
            identity_scope=identity,
            core_blocks=core_blocks,
            archival_memories=archival_memories,
        )
