from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from superbiz_agent.memory.runtime import MemoryRuntimeComponents
from superbiz_agent.memory.schemas import (
    CORE_BLOCK_SPECS,
    DEFAULT_CORE_BLOCK_KEYS,
    LongTermMemory,
    MemoryStatus,
    MemoryType,
    utc_now,
)
from superbiz_agent.memory.store import content_hash


@dataclass(frozen=True)
class ArchivalMemoryFixture:
    fixture_id: str
    topic: str
    content: str
    type: MemoryType = "experience"
    session_id: str | None = None
    tags: tuple[str, ...] = ()
    scope_service: str | None = None
    scope_env: str | None = None
    status: MemoryStatus = "active"
    usage_count: int = 0


@dataclass(frozen=True)
class CoreMemoryFixture:
    """Explicit initial Core Memory state for version-sensitive eval cases."""

    content: str
    version: int = 1

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("Core memory fixture version must be at least 1")


@dataclass(frozen=True)
class SeededMemoryFixtures:
    fixture_memory_ids: Mapping[str, str]


class MemoryFixtureSeeder:
    """Seed deterministic initial memory without invoking the agent or emitting trace events."""

    def __init__(self, memory_runtime: MemoryRuntimeComponents) -> None:
        self.memory_runtime = memory_runtime

    async def initialize_default_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> None:
        admin = self.memory_runtime.fixture_admin
        if admin is None:
            raise RuntimeError("memory fixtures are only available for the memory backend")
        existing = {
            block.block_key
            for block in await self.memory_runtime.inspection_repository.list_core_blocks_for_scope(
                tenant_id,
                user_id,
                agent_id,
            )
        }
        if any(block_key not in existing for block_key in DEFAULT_CORE_BLOCK_KEYS):
            await admin.ensure_default_core_blocks(tenant_id, user_id, agent_id)

    async def seed(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        core_blocks: Mapping[str, str | CoreMemoryFixture] | None = None,
        archival_memories: Sequence[ArchivalMemoryFixture] = (),
    ) -> SeededMemoryFixtures:
        await self.initialize_default_core_blocks(tenant_id, user_id, agent_id)
        admin = self.memory_runtime.fixture_admin
        if admin is None:
            raise RuntimeError("memory fixtures are only available for the memory backend")

        for block_key, fixture in (core_blocks or {}).items():
            if block_key not in CORE_BLOCK_SPECS:
                raise ValueError(f"Unknown core memory block: {block_key}")
            current = next(
                block
                for block in await (
                    self.memory_runtime.inspection_repository.list_core_blocks_for_scope(
                        tenant_id,
                        user_id,
                        agent_id,
                    )
                )
                if block.block_key == block_key
            )
            if isinstance(fixture, CoreMemoryFixture):
                await admin.upsert_core_block(
                    replace(
                        current,
                        content=fixture.content,
                        content_hash=content_hash(fixture.content),
                        version=fixture.version,
                    )
                )
                continue
            if not isinstance(fixture, str):
                raise TypeError("Core memory fixtures must be strings or CoreMemoryFixture")
            if current.content_hash != content_hash(fixture):
                await admin.upsert_core_block(
                    replace(
                        current,
                        content=fixture,
                        content_hash=content_hash(fixture),
                        version=current.version + 1,
                        updated_at=utc_now(),
                    )
                )

        fixture_memory_ids: dict[str, str] = {}
        for fixture in archival_memories:
            if not fixture.fixture_id.strip():
                raise ValueError("Archival memory fixture_id is required")
            if fixture.fixture_id in fixture_memory_ids:
                raise ValueError(f"Duplicate archival fixture_id: {fixture.fixture_id}")
            embedding = self.memory_runtime.embedding_service.embed(fixture.content)
            memory = LongTermMemory(
                tenant_id=tenant_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=fixture.session_id,
                type=fixture.type,
                topic=fixture.topic.strip(),
                content=fixture.content.strip(),
                embedding=embedding,
                content_hash=content_hash(fixture.content),
                source="eval_fixture",
                embedding_dimension=self.memory_runtime.embedding_service.dimension,
                usage_count=fixture.usage_count,
                status=fixture.status,
                tags=list(fixture.tags),
                scope_service=fixture.scope_service,
                scope_env=fixture.scope_env,
            )
            await admin.insert_memory(memory)
            fixture_memory_ids[fixture.fixture_id] = memory.id

        return SeededMemoryFixtures(fixture_memory_ids=dict(fixture_memory_ids))
