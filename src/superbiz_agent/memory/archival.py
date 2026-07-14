from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from superbiz_agent.harness.context import RunContext
from superbiz_agent.memory.embedding import DeterministicEmbeddingService
from superbiz_agent.memory.policy import MemoryWritePolicy, PolicyDecision
from superbiz_agent.memory.schemas import LongTermMemory
from superbiz_agent.memory.store import InMemoryMemoryStore


@dataclass(frozen=True)
class WriteMemoryResult:
    success: bool
    duplicate: bool = False
    memory_id: str | None = None
    existing_memory_id: str | None = None
    message: str = ""
    rejection: PolicyDecision | None = None
    dedupe_kind: Literal["exact"] | None = None
    metadata_merged: bool = False

    @classmethod
    def written(cls, memory_id: str) -> "WriteMemoryResult":
        return cls(
            success=True,
            memory_id=memory_id,
            message="Memory written",
            dedupe_kind="exact",
        )

    @classmethod
    def duplicate_skipped(
        cls,
        memory_id: str | None,
        existing_memory_id: str | None,
        *,
        metadata_merged: bool,
    ) -> "WriteMemoryResult":
        return cls(
            success=True,
            duplicate=True,
            memory_id=memory_id,
            existing_memory_id=existing_memory_id,
            message="Duplicate memory skipped",
            dedupe_kind="exact",
            metadata_merged=metadata_merged,
        )

    @classmethod
    def rejected(cls, decision: PolicyDecision) -> "WriteMemoryResult":
        message = f"Rejected: {decision.message}"
        if decision.suggestion:
            message += f" Suggestion: {decision.suggestion}"
        return cls(success=False, message=message, rejection=decision)

    @classmethod
    def failure(cls, message: str) -> "WriteMemoryResult":
        return cls(success=False, message=message)


class ArchivalMemoryService:
    def __init__(
        self,
        store: InMemoryMemoryStore,
        policy: MemoryWritePolicy,
        embedding_service: DeterministicEmbeddingService,
    ) -> None:
        self.store = store
        self.policy = policy
        self.embedding_service = embedding_service

    def save_archival_memory(
        self,
        run_context: RunContext,
        *,
        topic: str,
        content: str,
        evidence_summary: str | None = None,
        scope_service: str | None = None,
        scope_env: str | None = None,
        tags: list[str] | None = None,
    ) -> WriteMemoryResult:
        validation = self.policy.validate_archival(
            run_context,
            topic,
            content,
            evidence_summary,
        )
        if not validation.allowed:
            return WriteMemoryResult.rejected(validation.decision)

        request_context = run_context.request_context
        tenant_id = request_context.tenant_id or ""
        user_id = request_context.user_id or ""
        agent_id = request_context.agent_id or ""
        embedding = self.embedding_service.embed(content)
        memory = LongTermMemory(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=request_context.session_id,
            type="experience",
            topic=topic.strip(),
            content=content.strip(),
            embedding=embedding,
            content_hash=validation.content_hash or "",
            embedding_dimension=self.embedding_service.dimension,
            tags=_normalize_tags(tags),
            scope_service=_blank_to_none(scope_service),
            scope_env=_blank_to_none(scope_env),
        )
        write_result = self.store.write_archival_exact(memory)
        if write_result.status == "written":
            return WriteMemoryResult.written(write_result.memory.id)
        return WriteMemoryResult.duplicate_skipped(
            write_result.memory.id,
            write_result.memory.id,
            metadata_merged=write_result.metadata_merged,
        )


def _normalize_tags(tags: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags or []:
        value = tag.strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
