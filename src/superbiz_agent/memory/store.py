from __future__ import annotations

import hashlib
import threading
from copy import deepcopy
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import Literal

from superbiz_agent.memory.dedup import canonical_content_hash, stable_tag_union
from superbiz_agent.memory.errors import CoreMemoryContractError
from superbiz_agent.memory.ports import ExactMemoryWriteResult
from superbiz_agent.memory.schemas import (
    CORE_BLOCK_SPECS,
    DEFAULT_CORE_BLOCK_KEYS,
    CoreMemoryBlock,
    LongTermMemory,
    MemorySearchResult,
    MemoryTopicSummary,
    utc_now,
)

def content_hash(content: str | None) -> str:
    if content is None or not content.strip():
        return ""
    normalized = content.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class InMemoryMemoryStore:
    """Process-local memory store with tenant/user/agent isolation."""

    def __init__(self) -> None:
        self._core_blocks: dict[tuple[str, str, str, str], CoreMemoryBlock] = {}
        self._memories: dict[str, LongTermMemory] = {}
        self._lock = threading.Lock()

    def load_core_blocks(self, tenant_id: str, user_id: str, agent_id: str) -> list[CoreMemoryBlock]:
        with self._lock:
            return [
                block
                for key in DEFAULT_CORE_BLOCK_KEYS
                if (
                    block := self._core_blocks.get((tenant_id, user_id, agent_id, key))
                ) is not None
                and block.status == "active"
            ]

    def ensure_default_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[CoreMemoryBlock]:
        """Atomically initialize and return the three active default blocks."""

        with self._lock:
            existing = {
                block_key: self._core_blocks.get(
                    (tenant_id, user_id, agent_id, block_key)
                )
                for block_key in DEFAULT_CORE_BLOCK_KEYS
            }
            if any(
                block is not None
                and not _valid_existing_default_core_block(
                    block,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    block_key=block_key,
                )
                for block_key, block in existing.items()
            ):
                raise CoreMemoryContractError()

            blocks: list[CoreMemoryBlock] = []
            for block_key in DEFAULT_CORE_BLOCK_KEYS:
                key = (tenant_id, user_id, agent_id, block_key)
                block = existing[block_key]
                if block is None:
                    spec = CORE_BLOCK_SPECS[block_key]
                    block = CoreMemoryBlock(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        block_key=block_key,
                        description=spec.description,
                        max_tokens=spec.max_tokens,
                        content_hash=content_hash(""),
                    )
                    self._core_blocks[key] = block
                blocks.append(block)
            return blocks

    def list_core_blocks_for_scope(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[CoreMemoryBlock]:
        """Return defensive copies without lazily initializing missing blocks."""

        allowed_statuses = set(statuses) if statuses is not None else None
        key_order = {key: index for index, key in enumerate(DEFAULT_CORE_BLOCK_KEYS)}
        with self._lock:
            blocks = [
                deepcopy(block)
                for (tenant, user, agent, _block_key), block in self._core_blocks.items()
                if tenant == tenant_id
                and user == user_id
                and agent == agent_id
                and (allowed_statuses is None or block.status in allowed_statuses)
            ]
        return sorted(
            blocks,
            key=lambda block: (key_order.get(block.block_key, len(key_order)), block.block_key),
        )

    def upsert_core_block(self, block: CoreMemoryBlock) -> CoreMemoryBlock:
        with self._lock:
            self._core_blocks[
                (block.tenant_id, block.user_id, block.agent_id, block.block_key)
            ] = block
            return block

    def initialize_core_block(self, tenant_id: str, user_id: str, agent_id: str, block_key: str) -> CoreMemoryBlock:
        spec = CORE_BLOCK_SPECS[block_key]
        block = CoreMemoryBlock(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            block_key=block_key,
            description=spec.description,
            max_tokens=spec.max_tokens,
            content_hash=content_hash(""),
        )
        return self.upsert_core_block(block)

    def update_core_content(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        block_key: str,
        content: str,
    ) -> CoreMemoryBlock | None:
        with self._lock:
            key = (tenant_id, user_id, agent_id, block_key)
            current = self._core_blocks.get(key)
            if current is None:
                return None
            updated = replace(
                current,
                content=content,
                content_hash=content_hash(content),
                version=current.version + 1,
                updated_at=utc_now(),
            )
            self._core_blocks[key] = updated
            return updated

    def cas_replace_core_content(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        block_key: str,
        content: str,
        *,
        expected_version: int,
    ) -> tuple[
        Literal["updated", "unchanged", "conflict", "inactive", "read_only"],
        CoreMemoryBlock | None,
    ]:
        with self._lock:
            key = (tenant_id, user_id, agent_id, block_key)
            current = self._core_blocks.get(key)
            if current is None or current.status != "active":
                return "inactive", current
            if current.read_only:
                return "read_only", current
            target_hash = content_hash(content)
            if current.content_hash == target_hash:
                return "unchanged", current
            if current.version != expected_version:
                return "conflict", current
            updated = replace(
                current,
                content=content,
                content_hash=target_hash,
                version=current.version + 1,
                updated_at=utc_now(),
            )
            self._core_blocks[key] = updated
            return "updated", updated

    def insert_memory(self, memory: LongTermMemory) -> LongTermMemory:
        with self._lock:
            self._memories[memory.id] = memory
            return memory

    def write_archival_exact(self, memory: LongTermMemory) -> ExactMemoryWriteResult:
        """Atomically insert or merge tags for one active exact-equivalent memory."""

        if memory.status != "active":
            raise ValueError("write_archival_exact requires an active memory")
        canonical_hash = canonical_content_hash(memory.content)
        candidate = replace(
            memory,
            content_hash=canonical_hash,
            tags=stable_tag_union([], memory.tags),
        )
        candidate_key = _archival_exact_key(candidate)
        with self._lock:
            equivalents = [
                existing
                for existing in self._memories.values()
                if existing.status == "active"
                and _archival_exact_key(existing) == candidate_key
            ]
            if not equivalents:
                self._memories[candidate.id] = candidate
                return ExactMemoryWriteResult(
                    status="written",
                    memory=candidate,
                    metadata_merged=False,
                )

            existing = min(equivalents, key=lambda item: (item.created_at, item.id))
            merged_tags = stable_tag_union(existing.tags, candidate.tags)
            metadata_merged = merged_tags != existing.tags
            updated = replace(
                existing,
                content_hash=canonical_hash,
                tags=merged_tags,
                updated_at=utc_now() if metadata_merged else existing.updated_at,
            )
            self._memories[existing.id] = updated
            return ExactMemoryWriteResult(
                status="duplicate_skipped",
                memory=updated,
                metadata_merged=metadata_merged,
            )

    def active_memories(
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
        tag_set = {tag for tag in (tags or []) if tag}
        with self._lock:
            memories = [
                memory
                for memory in self._memories.values()
                if memory.status == "active"
                and memory.tenant_id == tenant_id
                and memory.user_id == user_id
                and memory.agent_id == agent_id
                and (not types or memory.type in types)
                and (not scope_service or memory.scope_service == scope_service)
                and (not scope_env or memory.scope_env == scope_env)
                and (not tag_set or any(tag in tag_set for tag in memory.tags))
            ]
        return sorted(memories, key=lambda item: (item.created_at, item.id))

    def list_memories_for_scope(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[LongTermMemory]:
        """Return defensive copies of memories for exactly one identity scope."""

        allowed_statuses = set(statuses) if statuses is not None else None
        with self._lock:
            memories = [
                deepcopy(memory)
                for memory in self._memories.values()
                if memory.tenant_id == tenant_id
                and memory.user_id == user_id
                and memory.agent_id == agent_id
                and (allowed_statuses is None or memory.status in allowed_statuses)
            ]
        return sorted(memories, key=lambda item: (item.created_at, item.id))

    def mark_returned(self, tenant_id: str, user_id: str, agent_id: str, memory_ids: list[str]) -> None:
        now = utc_now()
        id_set = set(memory_ids)
        with self._lock:
            for memory_id in id_set:
                memory = self._memories.get(memory_id)
                if (
                    memory is not None
                    and memory.tenant_id == tenant_id
                    and memory.user_id == user_id
                    and memory.agent_id == agent_id
                ):
                    self._memories[memory_id] = replace(
                        memory,
                        usage_count=memory.usage_count + 1,
                        last_used_at=now,
                        updated_at=now,
                    )

    def list_topics(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        types: list[str],
        limit_per_type: int,
    ) -> list[MemoryTopicSummary]:
        memories = self.active_memories(tenant_id, user_id, agent_id, types=types)
        summaries: list[MemoryTopicSummary] = []
        for memory_type in types:
            counter = Counter(memory.topic for memory in memories if memory.type == memory_type)
            for topic, count in counter.most_common(max(0, limit_per_type)):
                summaries.append(MemoryTopicSummary(type=memory_type, topic=topic, count=count))
        return summaries

    def all_tags(self, tenant_id: str, user_id: str, agent_id: str, limit: int = 10) -> list[str]:
        memories = self.active_memories(
            tenant_id,
            user_id,
            agent_id,
            types=["experience", "knowledge"],
        )
        counter = Counter(tag for memory in memories for tag in memory.tags)
        return [tag for tag, _count in counter.most_common(max(0, limit))]

    def all_scopes(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        limit: int = 10,
    ) -> tuple[list[str], list[str]]:
        memories = self.active_memories(
            tenant_id,
            user_id,
            agent_id,
            types=["experience", "knowledge"],
        )
        service_counter = Counter(
            memory.scope_service for memory in memories if memory.scope_service
        )
        env_counter = Counter(memory.scope_env for memory in memories if memory.scope_env)
        max_items = max(0, limit)
        services = [service for service, _count in service_counter.most_common(max_items)]
        envs = [env for env, _count in env_counter.most_common(max_items)]
        return services, envs

    def search_results_from_memories(
        self,
        memories: list[LongTermMemory],
        similarities: dict[str, float],
    ) -> list[MemorySearchResult]:
        return [
            MemorySearchResult(
                id=memory.id,
                type=memory.type,
                topic=memory.topic,
                content=memory.content,
                source=memory.source,
                similarity=similarities[memory.id],
                usage_count=memory.usage_count,
                last_used_at=memory.last_used_at,
                scope_service=memory.scope_service,
                scope_env=memory.scope_env,
                tags=list(memory.tags),
                confidence_label=confidence_label_for(similarities[memory.id]),
            )
            for memory in memories
        ]


def _archival_exact_key(
    memory: LongTermMemory,
) -> tuple[str, str, str, str, str | None, str | None, str]:
    return (
        memory.tenant_id,
        memory.user_id,
        memory.agent_id,
        memory.type,
        memory.scope_service,
        memory.scope_env,
        canonical_content_hash(memory.content),
    )


def _valid_existing_default_core_block(
    block: CoreMemoryBlock,
    *,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    block_key: str,
) -> bool:
    return (
        isinstance(block.id, str)
        and bool(block.id.strip())
        and block.tenant_id == tenant_id
        and block.user_id == user_id
        and block.agent_id == agent_id
        and block.block_key == block_key
        and block.status == "active"
        and isinstance(block.version, int)
        and not isinstance(block.version, bool)
        and block.version >= 1
        and isinstance(block.max_tokens, int)
        and not isinstance(block.max_tokens, bool)
        and block.max_tokens > 0
        and isinstance(block.content, str)
        and block.content_hash == content_hash(block.content)
    )


def confidence_label_for(similarity: float) -> str | None:
    if similarity >= 0.7:
        return "high"
    if similarity >= 0.5:
        return "medium"
    return None
