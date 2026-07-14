from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.stores import RolloutEventStore
from superbiz_agent.memory.archival import ArchivalMemoryService
from superbiz_agent.memory.core import CoreMemoryService
from superbiz_agent.memory.search import MemorySearchService
from superbiz_agent.tools.policies import DangerLevel, ToolPolicy
from superbiz_agent.tools.registry import ToolDefinition, ToolInvocationContext


MemoryTypeArg = Literal["experience", "knowledge", "all"]


class ListMemoryTopicsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: MemoryTypeArg = Field(
        default="all",
        description="Optional memory type filter for the returned topic routing summaries.",
    )


class SearchMemoryArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    query: str = Field(
        ...,
        min_length=1,
        description=(
            "Concise natural-language semantic query describing the actual historical "
            "experience, preference, or reusable knowledge to retrieve."
        ),
    )
    type: MemoryTypeArg = Field(
        default="all",
        description="Optional memory type filter; use all when the type is uncertain.",
    )
    scope_service: Optional[str] = Field(
        default=None,
        alias="scopeService",
        description=(
            "Optional case-sensitive exact service pre-filter. Use only an exact value from "
            "available memory scopes; omit when uncertain."
        ),
    )
    scope_env: Optional[str] = Field(
        default=None,
        alias="scopeEnv",
        description=(
            "Optional case-sensitive exact environment pre-filter. Use only an exact value "
            "from available memory scopes; omit when uncertain."
        ),
    )
    tags: Optional[str] = Field(
        default=None,
        description=(
            "Optional comma-separated, case-sensitive tag pre-filter with any-match "
            "semantics. Use only user-requested tags or exact available tags; ordinary query "
            "keywords, topics, services, and fault names are not tags."
        ),
    )


class UpdateCoreMemoryArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    block_key: str = Field(
        ...,
        alias="blockKey",
        min_length=1,
        description=(
            "Core block to replace: user_rules for durable interaction rules, "
            "user_ops_profile for stable user responsibilities/preferences, or service_notes "
            "for stable high-reuse service background."
        ),
    )
    new_content: str = Field(
        ...,
        alias="newContent",
        description=(
            "Complete replacement content for the selected block, not an incremental fragment. "
            "Preserve valid existing facts, merge duplicates, and remove obsolete conflicts."
        ),
    )
    change_reason: Optional[str] = Field(
        default=None,
        alias="changeReason",
        description="Optional concise reason for requesting this complete block replacement.",
    )


class SaveArchivalMemoryArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    topic: str = Field(
        ...,
        min_length=1,
        description="Concise routing topic for the verified reusable memory.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description=(
            "Self-contained, verified, reusable historical lesson or stable knowledge summary; "
            "never raw output, a secret, temporary state, or an unverified inference."
        ),
    )
    evidence_summary: Optional[str] = Field(
        default=None,
        alias="evidenceSummary",
        description="Optional concise description of the evidence that verified this memory.",
    )
    scope_service: Optional[str] = Field(
        default=None,
        alias="scopeService",
        description="Optional exact service metadata; omit rather than guess.",
    )
    scope_env: Optional[str] = Field(
        default=None,
        alias="scopeEnv",
        description="Optional exact environment metadata; omit rather than guess.",
    )
    tags: Optional[str] = Field(
        default=None,
        description=(
            "Optional comma-separated stable metadata tags derived only when clearly supported "
            "by the memory content; omit rather than guess."
        ),
    )


class MemoryToolHandlers:
    def __init__(
        self,
        *,
        core_service: CoreMemoryService,
        archival_service: ArchivalMemoryService,
        search_service: MemorySearchService,
        trace_store: RolloutEventStore,
    ) -> None:
        self.core_service = core_service
        self.archival_service = archival_service
        self.search_service = search_service
        self.trace_store = trace_store

    async def list_memory_topics(
        self,
        args: ListMemoryTopicsArgs,
        invocation_context: ToolInvocationContext,
    ) -> dict[str, Any]:
        run_context = invocation_context.run_context
        context = run_context.request_context
        topics = self.search_service.list_memory_topics(
            context.tenant_id or "",
            context.user_id or "",
            context.agent_id or "",
            args.type,
        )
        return {
            "success": True,
            "count": len(topics),
            "note": "Topics are only a routing aid; call searchMemory before using memory as evidence.",
            "topics": [
                {"type": topic.type, "topic": topic.topic, "count": topic.count}
                for topic in topics
            ],
        }

    async def search_memory(
        self,
        args: SearchMemoryArgs,
        invocation_context: ToolInvocationContext,
    ) -> dict[str, Any]:
        run_context = invocation_context.run_context
        context = run_context.request_context
        tag_list = parse_tags(args.tags)
        results = self.search_service.search_memory(
            context.tenant_id or "",
            context.user_id or "",
            context.agent_id or "",
            args.query,
            args.type,
            scope_service=args.scope_service,
            scope_env=args.scope_env,
            tags=tag_list,
        )
        await self.trace_store.append_event(
            run_context,
            RolloutEventType.MEMORY_SEARCHED,
            {
                "query": args.query,
                "type": args.type,
                "count": len(results),
                "scopeService": args.scope_service,
                "scopeEnv": args.scope_env,
                "tags": tag_list,
            },
        )
        return {
            "success": True,
            "count": len(results),
            "note": (
                "Historical memory is not current evidence; verify with realtime tools "
                "when diagnosing live issues."
            ),
            "memories": [
                {
                    "id": result.id,
                    "type": result.type,
                    "topic": result.topic,
                    "similarity": round(result.similarity, 4),
                    "confidenceLabel": result.confidence_label,
                    "content": result.content,
                }
                for result in results
            ],
        }

    async def update_core_memory(
        self,
        args: UpdateCoreMemoryArgs,
        invocation_context: ToolInvocationContext,
    ) -> dict[str, Any]:
        run_context = invocation_context.run_context
        result = self.core_service.update_block_with_policy(
            run_context,
            args.block_key,
            args.new_content,
            args.change_reason,
        )
        if result.success and result.block is not None and result.status == "updated":
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.CORE_MEMORY_UPDATED,
                {
                    "blockKey": result.block.block_key,
                    "version": result.block.version,
                    "status": "updated",
                },
            )
            return {
                "success": True,
                "status": "updated",
                "blockKey": result.block.block_key,
                "version": result.block.version,
                "message": "Core memory block updated.",
            }
        if result.success and result.block is not None and result.status == "unchanged":
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.CORE_MEMORY_UNCHANGED,
                {
                    "blockKey": result.block.block_key,
                    "version": result.block.version,
                    "status": "unchanged",
                },
            )
            return {
                "success": True,
                "status": "unchanged",
                "blockKey": result.block.block_key,
                "version": result.block.version,
                "message": "Core memory block already contains the same normalized content.",
            }
        rejection = result.rejection
        await self.trace_store.append_event(
            run_context,
            RolloutEventType.CORE_MEMORY_UPDATE_REJECTED,
            {
                "blockKey": args.block_key,
                "errorType": rejection.error_type if rejection else "unknown",
                "message": rejection.message if rejection else "Core memory update rejected.",
            },
        )
        return {
            "success": False,
            "status": "rejected",
            "error_type": rejection.error_type if rejection else "update_failed",
            "message": rejection.message if rejection else "Core memory update rejected.",
            "suggestion": rejection.suggestion if rejection else None,
        }

    async def save_archival_memory(
        self,
        args: SaveArchivalMemoryArgs,
        invocation_context: ToolInvocationContext,
    ) -> dict[str, Any]:
        run_context = invocation_context.run_context
        tag_list = parse_tags(args.tags)
        result = self.archival_service.save_archival_memory(
            run_context,
            topic=args.topic,
            content=args.content,
            evidence_summary=args.evidence_summary,
            scope_service=args.scope_service,
            scope_env=args.scope_env,
            tags=tag_list,
        )
        if result.success and not result.duplicate:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.ARCHIVAL_MEMORY_WRITTEN,
                _archival_payload(
                    args,
                    tag_list,
                    result.memory_id,
                    dedupe_kind=result.dedupe_kind,
                    metadata_merged=result.metadata_merged,
                ),
            )
            status = "written"
        elif result.success and result.duplicate:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.ARCHIVAL_MEMORY_DUPLICATE_SKIPPED,
                _archival_payload(
                    args,
                    tag_list,
                    result.existing_memory_id,
                    dedupe_kind=result.dedupe_kind,
                    metadata_merged=result.metadata_merged,
                ),
            )
            status = "duplicate_skipped"
        elif result.rejection is not None:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.MEMORY_WRITE_REJECTED,
                _archival_payload(
                    args,
                    tag_list,
                    None,
                    failure_kind=result.rejection.error_type,
                ),
            )
            status = "error"
        else:
            await self.trace_store.append_event(
                run_context,
                RolloutEventType.ARCHIVAL_MEMORY_WRITE_FAILED,
                _archival_payload(args, tag_list, None, failure_kind="write_failed"),
            )
            status = "error"
        return {
            "success": result.success,
            "status": status,
            "duplicate": result.duplicate,
            "memoryId": result.memory_id,
            "message": result.message,
            "note": "Long-term memory is historical reference, not current evidence.",
        }


def build_memory_tools(
    *,
    core_service: CoreMemoryService,
    archival_service: ArchivalMemoryService,
    search_service: MemorySearchService,
    trace_store: RolloutEventStore,
) -> list[ToolDefinition]:
    handlers = MemoryToolHandlers(
        core_service=core_service,
        archival_service=archival_service,
        search_service=search_service,
        trace_store=trace_store,
    )
    return [
        ToolDefinition(
            name="listMemoryTopics",
            description=(
                "List topic routing summaries for archival long-term memory. This tool returns "
                "topics, not tags or memory facts. Use it only when visible memory metadata is "
                "insufficient, and call searchMemory before using archival memory as evidence."
            ),
            args_model=ListMemoryTopicsArgs,
            handler=handlers.list_memory_topics,
            policy=ToolPolicy(
                tool_name="listMemoryTopics",
                danger_level=DangerLevel.LOW,
                timeout_seconds=2,
                max_retries=0,
                idempotent=True,
            ),
            requires_context=True,
        ),
        ToolDefinition(
            name="searchMemory",
            description=(
                "Search archival long-term memory. query is the natural-language semantic "
                "retrieval intent. tags is an optional comma-separated, case-sensitive any-match "
                "strict pre-filter; scopeService and scopeEnv are optional case-sensitive exact "
                "pre-filters. Use filters only when explicitly requested or exactly present in "
                "memory metadata; never turn ordinary keywords, topics, services, or fault names "
                "into guessed tags. Omit uncertain filters. Returned memory is historical "
                "reference, so verify live incidents with realtime tools."
            ),
            args_model=SearchMemoryArgs,
            handler=handlers.search_memory,
            policy=ToolPolicy(
                tool_name="searchMemory",
                danger_level=DangerLevel.LOW,
                timeout_seconds=4,
                max_retries=0,
                idempotent=True,
            ),
            requires_context=True,
        ),
        ToolDefinition(
            name="updateCoreMemory",
            description=(
                "Replace one always-visible Core Memory block only when both gates pass. First, "
                "durable intent is explicit through a direct save request or persistent framing "
                "such as long-term, fixed, always/future behavior, or an explicitly durable user "
                "responsibility or preference; an ordinary stable statement without durable "
                "framing is not authorization. Second, the content is clear, stable, verified, "
                "non-sensitive, and correctly routable. When both conditions are already clear, "
                "call this tool in the same turn without duplicate confirmation. Use user_rules, "
                "user_ops_profile, or service_notes according to their declared purposes. "
                "newContent is the complete replacement block. Never store raw output, temporary "
                "state, secrets, or unverified guesses, and never claim a write outcome before "
                "this tool returns. updated means this call wrote; unchanged means the content "
                "already existed and was not newly updated."
            ),
            args_model=UpdateCoreMemoryArgs,
            handler=handlers.update_core_memory,
            policy=ToolPolicy(
                tool_name="updateCoreMemory",
                danger_level=DangerLevel.MEDIUM,
                timeout_seconds=6,
                max_retries=0,
                idempotent=False,
            ),
            requires_context=True,
        ),
        ToolDefinition(
            name="saveArchivalMemory",
            description=(
                "Save searchable Archival Memory only when both gates pass. First, durable intent "
                "is explicit through a direct save request or persistent framing such as "
                "long-term, reusable knowledge, or knowledge that need not stay visible but must "
                "be saved for later retrieval; an ordinary stable statement without durable "
                "framing is not authorization. Second, the content is clear, verified, reusable, "
                "non-sensitive, and does not need to remain always visible. When both conditions "
                "are already clear, call this tool in the same turn without duplicate "
                "confirmation. Prefer it for verified incident lessons, root causes, runbook "
                "experience, and detailed historical knowledge. Never save raw output, temporary "
                "state, secrets, or unverified hypotheses. Add tags/scope only when clearly "
                "supported, and never claim a write outcome before the tool returns. written means "
                "this call saved; duplicate_skipped means it already existed."
            ),
            args_model=SaveArchivalMemoryArgs,
            handler=handlers.save_archival_memory,
            policy=ToolPolicy(
                tool_name="saveArchivalMemory",
                danger_level=DangerLevel.MEDIUM,
                timeout_seconds=6,
                max_retries=0,
                idempotent=False,
            ),
            requires_context=True,
        ),
    ]


def parse_tags(tags: str | None) -> list[str]:
    if tags is None or not tags.strip():
        return []
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _archival_payload(
    args: SaveArchivalMemoryArgs,
    tags: list[str],
    memory_id: str | None,
    *,
    dedupe_kind: str | None = None,
    metadata_merged: bool | None = None,
    failure_kind: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "topic": args.topic,
        "scopeService": args.scope_service,
        "scopeEnv": args.scope_env,
    }
    if tags:
        payload["tags"] = tags
    if memory_id is not None:
        payload["memoryId"] = memory_id
    if dedupe_kind is not None:
        payload["dedupeKind"] = dedupe_kind
    if metadata_merged is not None:
        payload["metadataMerged"] = metadata_merged
    if failure_kind is not None:
        payload["failureKind"] = failure_kind
    return payload
