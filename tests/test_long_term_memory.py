from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import pytest

from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.harness.context_assembler import ContextAssembler
from superbiz_agent.harness.events import RolloutEventType
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.memory.archival import ArchivalMemoryService
from superbiz_agent.memory.adapters.in_memory import InMemoryMemoryRepository
from superbiz_agent.memory.core import CoreMemoryService
from superbiz_agent.memory.dedup import (
    canonical_content_hash,
    canonicalize_archival_content,
)
from superbiz_agent.memory.embedding import DeterministicEmbeddingService
from superbiz_agent.memory.index import MemoryContextProvider, MemoryIndexService
from superbiz_agent.memory.policy import MemoryWritePolicy
from superbiz_agent.memory.run_snapshots import CoreVersionSnapshotRegistry
from superbiz_agent.memory.search import MemorySearchService
from superbiz_agent.memory.schemas import LongTermMemory
from superbiz_agent.memory.store import InMemoryMemoryStore
from superbiz_agent.model_gateway.base import ModelMessage, ModelToolCall
from superbiz_agent.model_gateway.openai_compatible import _tool_to_openai
from pathlib import Path

from superbiz_agent.prompts.registry import PromptRegistry
from superbiz_agent.security.permissions import LOCAL_DEFAULT_PERMISSIONS
from superbiz_agent.tools.errors import ToolErrorType


def _memory_services():
    store = InMemoryMemoryStore()
    repository = InMemoryMemoryRepository(store)
    policy = MemoryWritePolicy()
    embedding = DeterministicEmbeddingService()
    snapshots = CoreVersionSnapshotRegistry()
    core = CoreMemoryService(repository, policy, snapshots)
    search = MemorySearchService(repository, embedding)
    archival = ArchivalMemoryService(repository, policy, embedding)
    index = MemoryIndexService(repository, search)
    return store, policy, core, archival, search, index


def _run_context(
    tenant_id: str = "tenant",
    user_id: str = "user",
    agent_id: str = "agent",
    session_id: str = "session",
) -> RunContext:
    return RunContext(
        request_context=AgentRequestContext(
            tenant_id,
            user_id,
            agent_id,
            session_id,
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        ),
        run_id=str(uuid4()),
        prompt_version="ops-agent-system-v3",
        tool_schema_version="ops-tools-v3",
        model_provider="stub",
    )


def _memory_record(**overrides: object) -> LongTermMemory:
    content = str(overrides.pop("content", "order-service pool exhaustion resolved"))
    values: dict[str, object] = {
        "tenant_id": "tenant",
        "user_id": "user",
        "agent_id": "agent",
        "session_id": "session",
        "type": "experience",
        "topic": "original-topic",
        "content": content,
        "embedding": [0.0] * 64,
        "content_hash": canonical_content_hash(content),
        "tags": [],
        "scope_service": "order-service",
        "scope_env": "production",
    }
    values.update(overrides)
    return LongTermMemory(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_core_memory_defaults_xml_and_policy_rejections() -> None:
    _store, policy, core, _archival, _search, _index = _memory_services()
    blocks = await core.load_blocks("tenant", "user", "agent")

    assert [block.block_key for block in blocks] == [
        "user_rules",
        "user_ops_profile",
        "service_notes",
    ]
    assert [block.max_tokens for block in blocks] == [300, 500, 600]
    xml = core.build_context_block(blocks)
    assert xml.startswith("<core_memory>")
    assert 'key="user_rules"' in xml
    assert "max_tokens=300" in xml

    invalid = policy.validate_core_update(None, "bad", "content")
    assert invalid.allowed is False
    assert invalid.error_type == "missing_context"


@pytest.mark.asyncio
async def test_core_update_success_version_and_forbidden_content() -> None:
    service = AgentHarnessService.placeholder()
    gateway = service.graph.tool_gateway
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tm",
            "um",
            "am",
            "core-update",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    await service.runtime.assemble_context(run_context, "更新长期规则")

    updated = await gateway.execute(
        run_context,
        ModelToolCall(
            id="core-ok",
            name="updateCoreMemory",
            arguments={
                "blockKey": "user_rules",
                "newContent": "以后回答按现象、证据、判断、建议组织。",
                "changeReason": "user preference",
            },
        ),
    )
    updated_block = next(
        block
        for block in await service.memory_runtime.core_service.load_blocks("tm", "um", "am")
        if block.block_key == "user_rules"
    )
    unchanged = await gateway.execute(
        run_context,
        ModelToolCall(
            id="core-unchanged",
            name="updateCoreMemory",
            arguments={
                "blockKey": "user_rules",
                "newContent": "  以后回答按现象、证据、判断、建议组织。  ",
                "changeReason": "same content with normalized whitespace",
            },
        ),
    )
    unchanged_block = next(
        block
        for block in await service.memory_runtime.core_service.load_blocks("tm", "um", "am")
        if block.block_key == "user_rules"
    )
    invalid = await gateway.execute(
        run_context,
        ModelToolCall(
            id="core-invalid",
            name="updateCoreMemory",
            arguments={"blockKey": "bad", "newContent": "x"},
        ),
    )
    secret = await gateway.execute(
        run_context,
        ModelToolCall(
            id="core-secret",
            name="updateCoreMemory",
            arguments={"blockKey": "user_rules", "newContent": "api_key=abc123secret456"},
        ),
    )
    too_long = await gateway.execute(
        run_context,
        ModelToolCall(
            id="core-long",
            name="updateCoreMemory",
            arguments={"blockKey": "user_rules", "newContent": "x" * 1300},
        ),
    )

    assert updated.result["success"] is True
    assert updated.result["status"] == "updated"
    assert updated.result["version"] == 2
    assert unchanged.result["success"] is True
    assert unchanged.result["status"] == "unchanged"
    assert unchanged.result["version"] == 2
    assert unchanged_block.version == updated_block.version
    assert unchanged_block.updated_at == updated_block.updated_at
    assert invalid.result["error_type"] == "invalid_block_key"
    assert secret.result["error_type"] == "sensitive_data"
    assert too_long.result["error_type"] == "content_too_long"

    events = await service.trace_store.list_by_run("tm", run_context.run_id)
    updated_events = [
        event for event in events if event.event_type == RolloutEventType.CORE_MEMORY_UPDATED
    ]
    unchanged_events = [
        event for event in events if event.event_type == RolloutEventType.CORE_MEMORY_UNCHANGED
    ]
    assert len(updated_events) == 1
    assert len(unchanged_events) == 1
    assert updated_events[0].payload == {
        "blockKey": "user_rules",
        "version": 2,
        "status": "updated",
    }
    assert unchanged_events[0].payload == {
        "blockKey": "user_rules",
        "version": 2,
        "status": "unchanged",
    }


@pytest.mark.asyncio
async def test_archival_save_dedupe_search_filters_usage_and_index() -> None:
    service = AgentHarnessService.placeholder()
    gateway = service.graph.tool_gateway
    context = AgentRequestContext(
        "ta",
        "ua",
        "aa",
        "archival",
        permissions=LOCAL_DEFAULT_PERMISSIONS,
        auth_mode="dev_headers",
    )
    run_context = await service.runtime.start_run(context)

    save_args = {
        "topic": "order-service/5xx",
        "content": "order-service 5xx confirmed root cause was payment-service pool exhaustion.",
        "evidenceSummary": "confirmed by metrics",
        "scopeService": "order-service",
        "scopeEnv": "production",
        "tags": "order,5xx,payment",
    }
    saved = await gateway.execute(
        run_context,
        ModelToolCall(id="save", name="saveArchivalMemory", arguments=save_args),
    )
    duplicate = await gateway.execute(
        run_context,
        ModelToolCall(id="save-duplicate", name="saveArchivalMemory", arguments=save_args),
    )
    terminal_variant = await gateway.execute(
        run_context,
        ModelToolCall(
            id="save-terminal-variant",
            name="saveArchivalMemory",
            arguments={**save_args, "content": f"{save_args['content']}!"},
        ),
    )
    search = await gateway.execute(
        run_context,
        ModelToolCall(
            id="search",
            name="searchMemory",
            arguments={
                "query": "order-service payment pool 5xx",
                "type": "all",
                "scopeService": "order-service",
                "scopeEnv": "production",
                "tags": "payment",
            },
        ),
    )
    multi_tag_search = await gateway.execute(
        run_context,
        ModelToolCall(
            id="search-multi-tags-any-match",
            name="searchMemory",
            arguments={
                "query": "order-service payment pool 5xx",
                "type": "all",
                "scopeService": "order-service",
                "scopeEnv": "production",
                "tags": "missing,payment",
            },
        ),
    )
    topics = await gateway.execute(
        run_context,
        ModelToolCall(id="topics", name="listMemoryTopics", arguments={"type": "all"}),
    )

    assert saved.result["success"] is True
    assert saved.result["status"] == "written"
    assert duplicate.result["status"] == "duplicate_skipped"
    assert terminal_variant.result["status"] == "duplicate_skipped"
    assert search.result["success"] is True
    assert search.result["count"] == 1
    assert search.result["memories"][0]["confidenceLabel"] in {"medium", "high"}
    assert "payment-service" in search.result["memories"][0]["content"]
    assert multi_tag_search.result["success"] is True
    assert multi_tag_search.result["count"] == 1
    assert "payment-service" in multi_tag_search.result["memories"][0]["content"]
    assert topics.result["topics"][0]["topic"] == "order-service/5xx"

    metadata = await service.runtime.context_assembler.memory_context_provider.index_service.build_index(
        "ta",
        "ua",
        "aa",
    )
    assert metadata.startswith("<memory_metadata>")
    assert "- archival_memory_total: 1" in metadata
    assert "available_topics" in metadata
    assert "  - order-service/5xx (1)" in metadata
    assert "available_tags: order, 5xx, payment" in metadata
    assert "services: order-service" in metadata
    assert "envs: production" in metadata
    assert "Call searchMemory before using archival memory as evidence." in metadata

    events = await service.trace_store.list_by_run("ta", run_context.run_id)
    assert RolloutEventType.ARCHIVAL_MEMORY_WRITTEN in [event.event_type for event in events]
    assert RolloutEventType.ARCHIVAL_MEMORY_DUPLICATE_SKIPPED in [
        event.event_type for event in events
    ]
    assert RolloutEventType.MEMORY_SEARCHED in [event.event_type for event in events]


def test_canonical_archival_content_normalizes_only_approved_variations() -> None:
    left = "  Cafe\u0301\r\nincident\t resolved。。。  \r\n"
    right = "Café incident resolved"

    assert canonicalize_archival_content(left) == right
    assert canonical_content_hash(left) == canonical_content_hash(right)
    for suffix in (".", "！！！", "??", "……   "):
        assert canonical_content_hash(f"{right}{suffix}") == canonical_content_hash(
            right
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("service a,b", "service a:b"),
        ("limit=100", "limit=101"),
        ("do not retry", "do retry"),
        ("order-service failed", "payment-service failed"),
        ("path=/api/v1 items", "path=/api/v2 items"),
        ("configKey=Kafka", "configKey=kafka"),
        ("configKey=value;", "configKey=value"),
    ],
)
def test_canonical_archival_content_preserves_semantic_differences(
    left: str,
    right: str,
) -> None:
    assert canonical_content_hash(left) != canonical_content_hash(right)


def test_exact_dedupe_key_isolated_by_identity_type_and_scope() -> None:
    store = InMemoryMemoryStore()
    base = _memory_record()
    assert store.write_archival_exact(base).status == "written"

    variations = [
        replace(base, id=str(uuid4()), tenant_id="tenant-2"),
        replace(base, id=str(uuid4()), user_id="user-2"),
        replace(base, id=str(uuid4()), agent_id="agent-2"),
        replace(base, id=str(uuid4()), type="knowledge"),
        replace(base, id=str(uuid4()), scope_service="payment-service"),
        replace(base, id=str(uuid4()), scope_env="staging"),
    ]

    assert [store.write_archival_exact(item).status for item in variations] == [
        "written"
    ] * len(variations)


def test_archived_memory_does_not_block_new_active_exact_write() -> None:
    store = InMemoryMemoryStore()
    archived = replace(_memory_record(), status="archived")
    store.insert_memory(archived)

    active = replace(archived, id=str(uuid4()), status="active")
    result = store.write_archival_exact(active)
    memories = store.list_memories_for_scope("tenant", "user", "agent")

    assert result.status == "written"
    assert result.memory.id != archived.id
    assert sorted(memory.status for memory in memories) == ["active", "archived"]


def test_exact_duplicate_merges_tags_and_preserves_existing_record() -> None:
    store = InMemoryMemoryStore()
    original = _memory_record(tags=["first", "shared"])
    first = store.write_archival_exact(original)
    duplicate = replace(
        original,
        id=str(uuid4()),
        session_id="new-session",
        topic="new-topic",
        source="new-source",
        tags=["shared", "second", "first", "third"],
    )

    merged = store.write_archival_exact(duplicate)
    unchanged = store.write_archival_exact(replace(duplicate, id=str(uuid4())))
    stored = store.active_memories("tenant", "user", "agent")

    assert first.status == "written"
    assert merged.status == "duplicate_skipped"
    assert merged.metadata_merged is True
    assert unchanged.status == "duplicate_skipped"
    assert unchanged.metadata_merged is False
    assert len(stored) == 1
    assert stored[0].id == original.id
    assert stored[0].topic == original.topic
    assert stored[0].content == original.content
    assert stored[0].session_id == original.session_id
    assert stored[0].source == original.source
    assert stored[0].tags == ["first", "shared", "second", "third"]


def test_concurrent_exact_writes_create_only_one_active_memory() -> None:
    store = InMemoryMemoryStore()
    candidates = [
        replace(_memory_record(), id=str(uuid4()), tags=[f"tag-{index}"])
        for index in range(24)
    ]

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(store.write_archival_exact, candidates))

    assert [result.status for result in results].count("written") == 1
    assert [result.status for result in results].count("duplicate_skipped") == 23
    active = store.active_memories("tenant", "user", "agent")
    assert len(active) == 1
    assert set(active[0].tags) == {f"tag-{index}" for index in range(24)}


@pytest.mark.asyncio
async def test_semantically_similar_content_is_not_deduped_without_canonical_match() -> None:
    store, _policy, _core, archival, _search, _index = _memory_services()
    context = _run_context()
    first_content = (
        "order-service 5xx confirmed root cause was payment-service pool exhaustion"
    )
    second_content = (
        "payment-service pool exhaustion was confirmed root cause order-service 5xx"
    )
    embedding = DeterministicEmbeddingService()
    similarity = embedding.cosine_similarity(
        embedding.embed(first_content),
        embedding.embed(second_content),
    )

    first = await archival.save_archival_memory(
        context,
        topic="first-topic",
        content=first_content,
        scope_service="order-service",
    )
    second = await archival.save_archival_memory(
        context,
        topic="second-topic",
        content=second_content,
        scope_service="order-service",
    )

    assert similarity == pytest.approx(1.0)
    assert canonical_content_hash(first_content) != canonical_content_hash(second_content)
    assert first.duplicate is False
    assert second.duplicate is False
    assert len(store.active_memories("tenant", "user", "agent")) == 2


@pytest.mark.asyncio
async def test_exact_dedupe_trace_records_metadata_merge_without_internal_identity() -> None:
    service = AgentHarnessService.placeholder()
    gateway = service.graph.tool_gateway
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "trace-tenant",
            "trace-user",
            "trace-agent",
            "trace-session",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    first = await gateway.execute(
        run_context,
        ModelToolCall(
            id="trace-first",
            name="saveArchivalMemory",
            arguments={
                "topic": "original-topic",
                "content": "Café incident resolved",
                "tags": "first,shared",
            },
        ),
    )
    duplicate = await gateway.execute(
        run_context,
        ModelToolCall(
            id="trace-duplicate",
            name="saveArchivalMemory",
            arguments={
                "topic": "replacement-topic",
                "content": " Cafe\u0301\r\nincident   resolved！！！ ",
                "tags": "shared,second",
            },
        ),
    )
    events = await service.trace_store.list_by_run(
        "trace-tenant", run_context.run_id
    )
    written = next(
        event
        for event in events
        if event.event_type == RolloutEventType.ARCHIVAL_MEMORY_WRITTEN
    )
    skipped = next(
        event
        for event in events
        if event.event_type == RolloutEventType.ARCHIVAL_MEMORY_DUPLICATE_SKIPPED
    )
    stored = await service.memory_runtime.inspection_repository.list_memories_for_scope(
        "trace-tenant",
        "trace-user",
        "trace-agent",
        statuses=["active"],
    )

    assert first.result["status"] == "written"
    assert duplicate.result["status"] == "duplicate_skipped"
    assert written.payload["dedupeKind"] == "exact"
    assert written.payload["metadataMerged"] is False
    assert skipped.payload["dedupeKind"] == "exact"
    assert skipped.payload["metadataMerged"] is True
    assert stored[0].topic == "original-topic"
    assert stored[0].content == "Café incident resolved"
    assert stored[0].tags == ["first", "shared", "second"]
    forbidden_keys = {
        "contentHash",
        "content_hash",
        "tenantId",
        "userId",
        "agentId",
    }
    assert forbidden_keys.isdisjoint(written.payload)
    assert forbidden_keys.isdisjoint(skipped.payload)
    assert forbidden_keys.isdisjoint(first.result)
    assert forbidden_keys.isdisjoint(duplicate.result)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["!!!", "……", " \r\n 。  "])
async def test_archival_rejects_content_without_canonical_meaning(
    content: str,
) -> None:
    service = AgentHarnessService.placeholder()
    gateway = service.graph.tool_gateway
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "meaning-tenant",
            "meaning-user",
            "meaning-agent",
            "meaning-session",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )

    rejected = await gateway.execute(
        run_context,
        ModelToolCall(
            id="meaningless-content",
            name="saveArchivalMemory",
            arguments={"topic": "meaningless", "content": content},
        ),
    )
    events = await service.trace_store.list_by_run(
        "meaning-tenant", run_context.run_id
    )

    assert rejected.result["success"] is False
    assert rejected.result["status"] == "error"
    assert "cannot consist only of whitespace or sentence-ending punctuation" in (
        rejected.result["message"]
    )
    assert await service.memory_runtime.inspection_repository.list_memories_for_scope(
        "meaning-tenant",
        "meaning-user",
        "meaning-agent",
        statuses=["active"],
    ) == []
    assert RolloutEventType.ARCHIVAL_MEMORY_WRITTEN not in {
        event.event_type for event in events
    }
    rejection = next(
        event
        for event in events
        if event.event_type == RolloutEventType.MEMORY_WRITE_REJECTED
    )
    assert rejection.payload["failureKind"] == "empty_content"


@pytest.mark.asyncio
async def test_meaningful_terminal_punctuation_still_writes_and_exact_dedupes() -> None:
    service = AgentHarnessService.placeholder()
    gateway = service.graph.tool_gateway
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "valid-tenant",
            "valid-user",
            "valid-agent",
            "valid-session",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )

    written = await gateway.execute(
        run_context,
        ModelToolCall(
            id="meaningful-written",
            name="saveArchivalMemory",
            arguments={"topic": "valid", "content": "真实经验。"},
        ),
    )
    duplicate = await gateway.execute(
        run_context,
        ModelToolCall(
            id="meaningful-duplicate",
            name="saveArchivalMemory",
            arguments={"topic": "other-topic", "content": "真实经验"},
        ),
    )

    assert written.result["status"] == "written"
    assert duplicate.result["status"] == "duplicate_skipped"
    assert len(
        await service.memory_runtime.inspection_repository.list_memories_for_scope(
            "valid-tenant",
            "valid-user",
            "valid-agent",
            statuses=["active"],
        )
    ) == 1


@pytest.mark.asyncio
async def test_archival_policy_rejects_secret_and_extra_context_args() -> None:
    service = AgentHarnessService.placeholder()
    gateway = service.graph.tool_gateway
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tp",
            "up",
            "ap",
            "policy",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )

    secret = await gateway.execute(
        run_context,
        ModelToolCall(
            id="save-secret",
            name="saveArchivalMemory",
            arguments={
                "topic": "secret",
                "content": "password=super-secret",
            },
        ),
    )
    extra_context = await gateway.execute(
        run_context,
        ModelToolCall(
            id="search-extra",
            name="searchMemory",
            arguments={"query": "x", "tenantId": "spoof"},
        ),
    )

    assert secret.result["success"] is False
    assert "secrets" in secret.result["message"]
    assert extra_context.result["success"] is False
    assert extra_context.result["error_type"] == ToolErrorType.PARAM_VALIDATION_FAILED.value


def test_memory_tools_registered_schema_policy_and_requires_context() -> None:
    service = AgentHarnessService.placeholder()
    tools = {tool.name: tool for tool in service.graph.tool_gateway.registry.list()}

    for name in [
        "listMemoryTopics",
        "searchMemory",
        "updateCoreMemory",
        "saveArchivalMemory",
    ]:
        assert name in tools
        assert tools[name].requires_context is True
        assert tools[name].args_model.model_json_schema()["additionalProperties"] is False

    assert tools["listMemoryTopics"].policy.timeout_seconds == 2
    assert tools["searchMemory"].policy.timeout_seconds == 4
    assert tools["updateCoreMemory"].policy.timeout_seconds == 6
    assert tools["saveArchivalMemory"].policy.timeout_seconds == 6

    search_schema = tools["searchMemory"].args_model.model_json_schema()
    assert search_schema["required"] == ["query"]
    assert set(search_schema["properties"]) == {
        "query",
        "type",
        "scopeService",
        "scopeEnv",
        "tags",
    }
    assert "semantic query" in search_schema["properties"]["query"]["description"]
    assert "any-match" in search_schema["properties"]["tags"]["description"]
    assert "exact service pre-filter" in search_schema["properties"]["scopeService"][
        "description"
    ]
    assert "tenantId" not in search_schema["properties"]
    assert "runId" not in search_schema["properties"]

    core_schema = tools["updateCoreMemory"].args_model.model_json_schema()
    assert set(core_schema["properties"]) == {"blockKey", "newContent", "changeReason"}
    assert "Complete replacement" in core_schema["properties"]["newContent"]["description"]

    archival_schema = tools["saveArchivalMemory"].args_model.model_json_schema()
    assert set(archival_schema["properties"]) == {
        "topic",
        "content",
        "evidenceSummary",
        "scopeService",
        "scopeEnv",
        "tags",
    }
    assert "Self-contained" in archival_schema["properties"]["content"]["description"]
    assert "topics, not tags" in tools["listMemoryTopics"].description
    assert "any-match" in tools["searchMemory"].description
    core_description = tools["updateCoreMemory"].description
    archival_description = tools["saveArchivalMemory"].description
    assert "both gates pass" in core_description
    assert "long-term, fixed, always/future behavior" in core_description
    assert "durable user responsibility or preference" in core_description
    assert "ordinary stable statement without durable framing is not authorization" in (
        core_description
    )
    assert "both gates pass" in archival_description
    assert "long-term, reusable knowledge" in archival_description
    assert "need not stay visible but must be saved for later retrieval" in archival_description
    assert "ordinary stable statement without durable framing is not authorization" in (
        archival_description
    )
    assert "explicitly authorizes durable storage" not in core_description
    assert "explicitly authorizes durable storage" not in archival_description

    openai_search = _tool_to_openai(tools["searchMemory"])
    openai_properties = openai_search["function"]["parameters"]["properties"]
    assert openai_search["function"]["description"] == tools["searchMemory"].description
    assert "semantic query" in openai_properties["query"]["description"]
    assert "any-match" in openai_properties["tags"]["description"]


@pytest.mark.asyncio
async def test_context_assembler_injects_core_then_index_then_history_then_user() -> None:
    _store, _policy, core, _archival, search, index = _memory_services()
    snapshots = core.snapshots
    provider = MemoryContextProvider(core, index, snapshots)
    assembler = ContextAssembler(
        PromptRegistry(Path(__file__).resolve().parents[1] / "prompts"),
        memory_context_provider=provider,
    )
    run_context = _run_context("tc", "uc", "ac", "sc")
    assembled = await assembler.assemble(
        prompt_version="ops-agent-system-v2",
        active_history=[ModelMessage(role="assistant", content="历史回答")],
        current_user_message="当前问题",
        run_context=run_context,
    )

    assert [message.role for message in assembled.messages] == [
        "system",
        "system",
        "system",
        "assistant",
        "user",
    ]
    assert assembled.messages[1].content.startswith("<core_memory>")
    assert assembled.messages[2].content.startswith("<memory_metadata>")
    assert "- available_tags: none" in assembled.messages[2].content
    assert "- available_scopes:" in assembled.messages[2].content
    assert assembled.trace_payload["hasCoreMemory"] is True
    assert assembled.trace_payload["hasMemoryIndex"] is True
    assert assembled.trace_payload["hasMemoryMetadata"] is True
    assert assembled.trace_payload["coreMemoryBlockCount"] == 3
