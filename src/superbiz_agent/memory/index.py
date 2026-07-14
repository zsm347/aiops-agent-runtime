from __future__ import annotations

from superbiz_agent.memory.policy import estimate_tokens
from superbiz_agent.memory.schemas import MemoryContext
from superbiz_agent.memory.search import MemorySearchService
from superbiz_agent.memory.store import InMemoryMemoryStore


class MemoryIndexService:
    def __init__(
        self,
        store: InMemoryMemoryStore,
        search_service: MemorySearchService,
        *,
        max_tokens: int = 600,
        topic_max_items_per_type: int = 5,
    ) -> None:
        self.store = store
        self.search_service = search_service
        self.max_tokens = max_tokens
        self.topic_max_items_per_type = topic_max_items_per_type

    def build_index(self, tenant_id: str, user_id: str, agent_id: str) -> str:
        memories = self.store.active_memories(
            tenant_id,
            user_id,
            agent_id,
            types=["experience", "knowledge"],
        )
        archival_total = len(memories)
        top_topics = self.search_service.list_memory_topics(tenant_id, user_id, agent_id, "all")
        tags = self.store.all_tags(tenant_id, user_id, agent_id, 10)
        services, envs = self.store.all_scopes(tenant_id, user_id, agent_id, 10)
        lines = [
            "<memory_metadata>",
            f"- archival_memory_total: {archival_total}",
            "- available_topics:",
        ]
        if top_topics:
            for memory_type in ("experience", "knowledge"):
                lines.append(f"  {memory_type}:")
                rendered = 0
                for topic in top_topics:
                    if topic.type != memory_type or rendered >= self.topic_max_items_per_type:
                        continue
                    lines.append(f"  - {topic.topic} ({topic.count})")
                    rendered += 1
        else:
            lines.append("  none")
        lines.append(f"- available_tags: {', '.join(tags) if tags else 'none'}")
        lines.append("- available_scopes:")
        lines.append(f"  services: {', '.join(services) if services else 'none'}")
        lines.append(f"  envs: {', '.join(envs) if envs else 'none'}")
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
            return self._fallback_within_budget()
        return rendered

    def count_index_topics(self, tenant_id: str, user_id: str, agent_id: str) -> int:
        return len(self.search_service.list_memory_topics(tenant_id, user_id, agent_id, "all"))

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
    def __init__(self, core_service, index_service: MemoryIndexService) -> None:
        self.core_service = core_service
        self.index_service = index_service

    def build_context(self, tenant_id: str, user_id: str, agent_id: str) -> MemoryContext:
        core_xml = self.core_service.build_context_block(tenant_id, user_id, agent_id)
        index_xml = self.index_service.build_index(tenant_id, user_id, agent_id)
        core_block_count = len(self.core_service.load_blocks(tenant_id, user_id, agent_id))
        topic_count = self.index_service.count_index_topics(tenant_id, user_id, agent_id)
        return MemoryContext(
            core_memory_xml=core_xml,
            memory_index_xml=index_xml,
            core_block_count=core_block_count,
            memory_index_topic_count=topic_count,
        )
