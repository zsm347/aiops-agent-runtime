from __future__ import annotations

from collections import Counter
import logging
import math

from superbiz_agent.memory.embedding import MemoryEmbeddingService
from superbiz_agent.memory.errors import MemoryStoreContractError
from superbiz_agent.memory.ports import MemoryRepository
from superbiz_agent.memory.schemas import MemorySearchResult, MemoryTopicSummary
from superbiz_agent.memory.store import confidence_label_for


SEARCHABLE_TYPES = ["experience", "knowledge"]
logger = logging.getLogger(__name__)


class MemorySearchService:
    def __init__(
        self,
        repository: MemoryRepository,
        embedding_service: MemoryEmbeddingService,
        *,
        top_k: int = 3,
        min_similarity: float = 0.5,
        topics_max: int = 30,
        topics_per_type: int = 20,
    ) -> None:
        self.repository = repository
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.topics_max = topics_max
        self.topics_per_type = topics_per_type

    async def search_memory(
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
        query_embedding = await self.embedding_service.embed_query(query)
        threshold = self.min_similarity if min_similarity is None else min_similarity
        hits = await self.repository.search_active_memories_by_vector(
            tenant_id,
            user_id,
            agent_id,
            query_embedding.vector,
            query_embedding.identity,
            types=types,
            scope_service=_blank_to_none(scope_service),
            scope_env=_blank_to_none(scope_env),
            tags=tags,
            min_similarity=threshold,
            limit=max(0, self.top_k),
        )
        _validate_hits(
            hits,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            types=types,
            scope_service=_blank_to_none(scope_service),
            scope_env=_blank_to_none(scope_env),
            tags=tags,
            min_similarity=threshold,
            limit=max(0, self.top_k),
            embedding_identity=query_embedding.identity,
        )
        results = [
            MemorySearchResult(
                id=hit.memory.id,
                type=hit.memory.type,
                topic=hit.memory.topic,
                content=hit.memory.content,
                source=hit.memory.source,
                similarity=hit.similarity,
                usage_count=hit.memory.usage_count,
                last_used_at=hit.memory.last_used_at,
                scope_service=hit.memory.scope_service,
                scope_env=hit.memory.scope_env,
                tags=list(hit.memory.tags),
                confidence_label=confidence_label_for(hit.similarity),
            )
            for hit in hits
        ]
        try:
            await self.repository.mark_returned(
                tenant_id,
                user_id,
                agent_id,
                [result.id for result in results],
            )
        except Exception:
            logger.warning("memory_usage_update_failed")
        return results

    async def list_memory_topics(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        optional_type: str | None = None,
    ) -> list[MemoryTopicSummary]:
        types = searchable_types(optional_type)
        if not types:
            return []
        memories = await self.repository.list_active_memories(
            tenant_id,
            user_id,
            agent_id,
            types=types,
        )
        topics: list[MemoryTopicSummary] = []
        for memory_type in types:
            counts = Counter(memory.topic for memory in memories if memory.type == memory_type)
            topics.extend(
                MemoryTopicSummary(type=memory_type, topic=topic, count=count)
                for topic, count in counts.most_common(max(0, self.topics_per_type))
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


def _validate_hits(
    hits,
    *,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    types: list[str],
    scope_service: str | None,
    scope_env: str | None,
    tags: list[str] | None,
    min_similarity: float,
    limit: int,
    embedding_identity,
) -> None:
    if len(hits) > limit:
        raise MemoryStoreContractError()
    tag_set = {tag for tag in (tags or []) if isinstance(tag, str) and tag}
    previous: tuple[float, str] | None = None
    for hit in hits:
        memory = hit.memory
        current = (-hit.similarity, memory.id)
        if (
            memory.tenant_id != tenant_id
            or memory.user_id != user_id
            or memory.agent_id != agent_id
            or memory.status != "active"
            or memory.type not in types
            or (scope_service is not None and memory.scope_service != scope_service)
            or (scope_env is not None and memory.scope_env != scope_env)
            or (tag_set and not tag_set.intersection(memory.tags))
            or memory.embedding_provider != embedding_identity.provider
            or memory.embedding_model != embedding_identity.model
            or memory.embedding_version != embedding_identity.version
            or memory.embedding_dimension != embedding_identity.dimension
            or memory.embedding_metric != "cosine"
            or not math.isfinite(hit.similarity)
            or hit.similarity < min_similarity
            or (previous is not None and current < previous)
        ):
            raise MemoryStoreContractError()
        previous = current
