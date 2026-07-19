from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


MemoryType = Literal["rule", "experience", "knowledge"]
MemoryStatus = Literal["active", "archived"]

USER_RULES = "user_rules"
USER_OPS_PROFILE = "user_ops_profile"
SERVICE_NOTES = "service_notes"
DEFAULT_CORE_BLOCK_KEYS = (USER_RULES, USER_OPS_PROFILE, SERVICE_NOTES)


@dataclass(frozen=True)
class CoreBlockSpec:
    description: str
    max_tokens: int


CORE_BLOCK_SPECS: dict[str, CoreBlockSpec] = {
    USER_RULES: CoreBlockSpec(
        description=(
            "Stable user rules to follow across sessions, such as response format, "
            "long-term constraints, or forbidden suggestions. Store only explicit durable "
            "preferences; do not store one-time task state, current alerts/logs/metrics, "
            "unverified guesses, service runtime facts, or secrets. When updating, provide "
            "a compact full block that preserves useful rules, merges duplicates, and removes "
            "outdated or conflicting rules."
        ),
        max_tokens=300,
    ),
    USER_OPS_PROFILE: CoreBlockSpec(
        description=(
            "Stable operational background about the user, including long-term owned "
            "services, common environments, troubleshooting habits, and tool/platform "
            "preferences. Do not store temporary incident roots, realtime logs, one-time "
            "query conditions, or service names unless they are confirmed as durable user "
            "context."
        ),
        max_tokens=500,
    ),
    SERVICE_NOTES: CoreBlockSpec(
        description=(
            "Stable, high-reuse service background such as common dependencies, architecture "
            "facts, and troubleshooting entry points. Do not store unverified root causes, "
            "current metric values, raw traces/logs, one-time change state, or detailed "
            "incident lessons; save detailed historical lessons with saveArchivalMemory."
        ),
        max_tokens=600,
    ),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CoreMemoryBlock:
    tenant_id: str
    user_id: str
    agent_id: str
    block_key: str
    description: str
    content: str = ""
    max_tokens: int = 500
    version: int = 1
    read_only: bool = False
    source: str = "memory_service"
    content_hash: str = ""
    status: MemoryStatus = "active"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class LongTermMemory:
    tenant_id: str
    user_id: str
    agent_id: str
    session_id: str | None
    type: MemoryType
    topic: str
    content: str
    embedding: list[float]
    content_hash: str
    source: str = "realtime"
    embedding_provider: str = "local-deterministic"
    embedding_model: str = "local-deterministic"
    embedding_dimension: int = 64
    embedding_metric: str = "cosine"
    embedding_version: str = "phase4-local"
    usage_count: int = 0
    last_used_at: datetime | None = None
    status: MemoryStatus = "active"
    archived_at: datetime | None = None
    archive_reason: str | None = None
    tags: list[str] = field(default_factory=list)
    scope_service: str | None = None
    scope_env: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class MemorySearchResult:
    id: str
    type: str
    topic: str
    content: str
    source: str
    similarity: float
    usage_count: int
    last_used_at: datetime | None = None
    scope_service: str | None = None
    scope_env: str | None = None
    tags: list[str] = field(default_factory=list)
    confidence_label: str | None = None


@dataclass(frozen=True)
class MemoryTopicSummary:
    type: str
    topic: str
    count: int


@dataclass(frozen=True)
class MemoryContext:
    core_memory_xml: str
    memory_index_xml: str
    core_block_count: int
    memory_index_topic_count: int

    @property
    def has_core_memory(self) -> bool:
        return bool(self.core_memory_xml.strip())

    @property
    def has_memory_index(self) -> bool:
        return bool(self.memory_index_xml.strip())
