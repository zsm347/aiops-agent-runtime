from __future__ import annotations

from superbiz_agent.memory.embedding import DeterministicEmbeddingService
from superbiz_agent.memory.schemas import MemorySearchResult, MemoryTopicSummary
from superbiz_agent.memory.store import InMemoryMemoryStore


SEARCHABLE_TYPES = ["experience", "knowledge"]


class MemorySearchService:
    def __init__(
        self,
        store: InMemoryMemoryStore,
        embedding_service: DeterministicEmbeddingService,
        *,
        top_k: int = 3,
        min_similarity: float = 0.5,
        topics_max: int = 30,
        topics_per_type: int = 20,
    ) -> None:
        self.store = store
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.topics_max = topics_max
        self.topics_per_type = topics_per_type

    def search_memory(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        query: str,
        optional_type: str | None = None,
        *,
        scope_service: str | None = None,
        scope_env: str | None = None,
        tags: list[str] | None = None,
        min_similarity: float | None = None,
    ) -> list[MemorySearchResult]:
        types = searchable_types(optional_type)
        if not types:
            return []
        query_embedding = self.embedding_service.embed(query)
        memories = self.store.active_memories(
            tenant_id,
            user_id,
            agent_id,
            types=types,
            scope_service=_blank_to_none(scope_service),
            scope_env=_blank_to_none(scope_env),
            tags=tags,
        )
        similarities = {
            memory.id: self.embedding_service.cosine_similarity(query_embedding, memory.embedding)
            for memory in memories
        }
        threshold = self.min_similarity if min_similarity is None else min_similarity
        selected_memories = [
            memory for memory in memories if similarities.get(memory.id, 0.0) >= threshold
        ]
        selected_memories.sort(key=lambda memory: (-similarities[memory.id], memory.id))
        selected_memories = selected_memories[: max(0, self.top_k)]
        results = self.store.search_results_from_memories(selected_memories, similarities)
        self.store.mark_returned(tenant_id, user_id, agent_id, [result.id for result in results])
        return results

    def list_memory_topics(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        optional_type: str | None = None,
    ) -> list[MemoryTopicSummary]:
        types = searchable_types(optional_type)
        if not types:
            return []
        topics = self.store.list_topics(
            tenant_id,
            user_id,
            agent_id,
            types,
            self.topics_per_type,
        )
        return topics[: max(0, self.topics_max)]


def searchable_types(optional_type: str | None) -> list[str]:
    if optional_type is None or not optional_type.strip():
        return list(SEARCHABLE_TYPES)
    normalized = optional_type.strip().lower()
    if normalized == "all":
        return list(SEARCHABLE_TYPES)
    if normalized in SEARCHABLE_TYPES:
        return [normalized]
    return []


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
