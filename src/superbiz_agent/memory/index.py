from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from xml.sax.saxutils import escape

from superbiz_agent.harness.context import RunContext
from superbiz_agent.memory.policy import estimate_tokens
from superbiz_agent.memory.ports import MemoryRepository
from superbiz_agent.memory.run_snapshots import CoreVersionSnapshotRegistry
from superbiz_agent.memory.schemas import MemoryContext
from superbiz_agent.memory.search import MemorySearchService


@dataclass(frozen=True)
class MemoryIndexBuild:
    xml: str
    topic_count: int


class MemoryIndexService:
    def __init__(
        self,
        repository: MemoryRepository,
        search_service: MemorySearchService,
        *,
        max_tokens: int = 600,
        topic_max_items_per_type: int = 5,
    ) -> None:
        self.repository = repository
        self.search_service = search_service
        self.max_tokens = max_tokens
        self.topic_max_items_per_type = topic_max_items_per_type

    async def build_index(self, tenant_id: str, user_id: str, agent_id: str) -> str:
        return (await self.build_index_with_count(tenant_id, user_id, agent_id)).xml

    async def build_index_with_count(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> MemoryIndexBuild:
        memories = await self.repository.list_active_memories(
            tenant_id,
            user_id,
            agent_id,
            types=["experience", "knowledge"],
        )
        archival_total = len(memories)
        top_topics = []
        for memory_type in ("experience", "knowledge"):
            counts = Counter(memory.topic for memory in memories if memory.type == memory_type)
            top_topics.extend(
                (memory_type, topic, count)
                for topic, count in counts.most_common(self.search_service.topics_per_type)
            )
        top_topics = top_topics[: self.search_service.topics_max]
        tag_counts = Counter(tag for memory in memories for tag in memory.tags)
        tags = [tag for tag, _count in tag_counts.most_common(10)]
        service_counts = Counter(
            memory.scope_service for memory in memories if memory.scope_service
        )
        env_counts = Counter(memory.scope_env for memory in memories if memory.scope_env)
        services = [service for service, _count in service_counts.most_common(10)]
        envs = [env for env, _count in env_counts.most_common(10)]
        lines = [
            "<memory_metadata>",
            f"- archival_memory_total: {archival_total}",
            "- available_topics:",
        ]
        if top_topics:
            for memory_type in ("experience", "knowledge"):
                lines.append(f"  {memory_type}:")
                rendered = 0
                for topic_type, topic, count in top_topics:
                    if topic_type != memory_type or rendered >= self.topic_max_items_per_type:
                        continue
                    lines.append(f"  - {escape(topic)} ({count})")
                    rendered += 1
        else:
            lines.append("  none")
        rendered_tags = ", ".join(escape(tag) for tag in tags) if tags else "none"
        lines.append(f"- available_tags: {rendered_tags}")
        lines.append("- available_scopes:")
        rendered_services = ", ".join(escape(service) for service in services) if services else "none"
        rendered_envs = ", ".join(escape(env) for env in envs) if envs else "none"
        lines.append(f"  services: {rendered_services}")
        lines.append(f"  envs: {rendered_envs}")
        lines.extend(
            [
                "- usage_rules:",
                "  - Use this metadata only to decide whether memory lookup may help.",
                "  - Do not answer concrete facts from metadata alone.",
                "  - Call searchMemory before using archival memory as evidence.",
                "  - Historical memory is reference only; verify live incidents with realtime tools.",
                "  - Tags and scopes are optional filters. Use them only when clear.",
                "</memory_metadata>",
            ]
        )
        rendered = "\n".join(lines)
        if self.max_tokens > 0 and estimate_tokens(rendered) > self.max_tokens:
            rendered = self._fallback_within_budget()
        return MemoryIndexBuild(xml=rendered, topic_count=len(top_topics))

    async def count_index_topics(self, tenant_id: str, user_id: str, agent_id: str) -> int:
        return (await self.build_index_with_count(tenant_id, user_id, agent_id)).topic_count

    def _fallback_within_budget(self) -> str:
        fallbacks = [
            "<memory_metadata>\nThis metadata is partial; call listMemoryTopics when needed.\n"
            "Call searchMemory before using archival memory as evidence.\n"
            "Historical memory is reference only, not current evidence.\n</memory_metadata>",
            "<memory_metadata>\nHistorical memory is reference only, not current evidence.\n</memory_metadata>",
            "<memory_metadata>\n</memory_metadata>",
            "",
        ]
        for fallback in fallbacks:
            if self.max_tokens <= 0 or estimate_tokens(fallback) <= self.max_tokens:
                return fallback
        return ""


class MemoryContextProvider:
    def __init__(
        self,
        core_service,
        index_service: MemoryIndexService,
        snapshots: CoreVersionSnapshotRegistry,
    ) -> None:
        self.core_service = core_service
        self.index_service = index_service
        self.snapshots = snapshots

    async def build_context(self, run_context: RunContext) -> MemoryContext:
        request_context = run_context.request_context
        tenant_id = request_context.tenant_id or ""
        user_id = request_context.user_id or ""
        agent_id = request_context.agent_id or ""
        blocks = await self.core_service.load_blocks(tenant_id, user_id, agent_id)
        core_xml = self.core_service.build_context_block(blocks)
        index = await self.index_service.build_index_with_count(tenant_id, user_id, agent_id)
        context = MemoryContext(
            core_memory_xml=core_xml,
            memory_index_xml=index.xml,
            core_block_count=len(blocks),
            memory_index_topic_count=index.topic_count,
        )
        self.snapshots.capture_once(run_context, blocks)
        return context
