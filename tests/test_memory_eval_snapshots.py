from __future__ import annotations

from dataclasses import replace

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.evals.memory_fixtures import (
    ArchivalMemoryFixture,
    CoreMemoryFixture,
    MemoryFixtureSeeder,
)
from superbiz_agent.evals.memory_snapshots import MemorySnapshotProvider
from superbiz_agent.harness.context import AgentRequestContext, RunContext
from superbiz_agent.memory.errors import CoreMemoryContractError, MemoryStoreIsolationError
from superbiz_agent.memory.adapters.in_memory import InMemoryMemoryRepository
from superbiz_agent.memory.schemas import CORE_BLOCK_SPECS, CoreMemoryBlock
from superbiz_agent.memory.store import content_hash
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore
from superbiz_agent.memory.runtime import build_memory_runtime
from superbiz_agent.model_gateway.base import ModelToolCall
from superbiz_agent.model_gateway.stub import StubModelGateway
from superbiz_agent.security.permissions import LOCAL_DEFAULT_PERMISSIONS


def _settings(*, memory_enabled: bool = True) -> Settings:
    return Settings(
        memory_enabled=memory_enabled,
        model_provider="stub",
        rollout_store_backend="memory",
        context_content_compression_backend="deterministic",
        _env_file=None,
    )


def _runtime():
    trace_store = InMemoryRolloutEventStore()
    runtime = build_memory_runtime(_settings(), trace_store=trace_store)
    return trace_store, runtime


@pytest.mark.asyncio
async def test_snapshot_is_read_only_and_does_not_lazy_initialize_core_blocks() -> None:
    trace_store, runtime = _runtime()
    provider = MemorySnapshotProvider(runtime.inspection_repository)

    first = await provider.capture("tenant", "user", "agent")
    second = await provider.capture("tenant", "user", "agent")

    assert first == second
    assert first.core_blocks == ()
    assert first.archival_memories == ()
    assert await runtime.inspection_repository.list_core_blocks_for_scope(
        "tenant", "user", "agent"
    ) == []
    assert await trace_store.list_by_session(
        AgentRequestContext("tenant", "user", "agent", "snapshot")
    ) == []


@pytest.mark.asyncio
async def test_fixture_initialization_precedes_snapshot_and_is_not_an_agent_write() -> None:
    trace_store, runtime = _runtime()
    seeder = MemoryFixtureSeeder(runtime)
    provider = MemorySnapshotProvider(runtime.inspection_repository)

    seeded = await seeder.seed(
        "tenant",
        "user",
        "agent",
        core_blocks={"user_rules": "回答时先给证据。"},
        archival_memories=[
            ArchivalMemoryFixture(
                fixture_id="mem-order",
                topic="order-service/5xx",
                content="order-service 历史 5xx 根因是连接池耗尽。",
                tags=("order-service", "5xx"),
                scope_service="order-service",
                scope_env="production",
            )
        ],
    )
    before = await provider.capture("tenant", "user", "agent")

    assert [block.block_key for block in before.core_blocks] == [
        "user_rules",
        "user_ops_profile",
        "service_notes",
    ]
    assert before.core_blocks[0].content == "回答时先给证据。"
    assert before.core_blocks[0].version == 2
    assert before.archival_memories[0].id == seeded.fixture_memory_ids["mem-order"]
    assert before.archival_memories[0].tags == ("order-service", "5xx")
    assert await trace_store.list_by_session(
        AgentRequestContext("tenant", "user", "agent", "fixture")
    ) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("block_key", "initial_content", "initial_version", "next_content", "expected_version_delta"),
    [
        ("user_rules", "回答先给证据，再给结论。", 3, "回答先给证据，再给结论。", 0),
        ("user_rules", "回答不要使用表格；结论放在最后。", 2, "优先使用表格；结论放在最后。", 1),
        (
            "user_ops_profile",
            "长期负责 order-service 和 inventory-service。",
            4,
            "长期负责 order-service 和 checkout-service。",
            1,
        ),
        (
            "service_notes",
            "order-service 同步依赖 legacy-inventory。",
            2,
            "order-service 同步依赖 inventory-v2。",
            1,
        ),
    ],
)
async def test_explicit_core_fixture_version_establishes_expected_version_delta(
    block_key: str,
    initial_content: str,
    initial_version: int,
    next_content: str,
    expected_version_delta: int,
) -> None:
    trace_store, runtime = _runtime()
    seeder = MemoryFixtureSeeder(runtime)
    provider = MemorySnapshotProvider(runtime.inspection_repository)
    await seeder.seed(
        "tenant",
        "user",
        "agent",
        core_blocks={
            block_key: CoreMemoryFixture(content=initial_content, version=initial_version)
        },
    )
    before = await provider.capture("tenant", "user", "agent")
    before_block = next(block for block in before.core_blocks if block.block_key == block_key)
    service = AgentHarnessService.build_default(
        _settings(),
        model_gateway=StubModelGateway(),
        trace_store=trace_store,
        memory_runtime=runtime,
    )
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tenant",
            "user",
            "agent",
            "version-delta",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    await service.runtime.assemble_context(run_context, "更新 Core Memory")
    result = await service.graph.tool_gateway.execute(
        run_context,
        ModelToolCall(
            id=f"update-{block_key}",
            name="updateCoreMemory",
            arguments={"blockKey": block_key, "newContent": next_content},
        ),
    )
    after = await provider.capture("tenant", "user", "agent")
    after_block = next(block for block in after.core_blocks if block.block_key == block_key)

    assert before_block.version == initial_version
    assert result.result["success"] is True
    assert after_block.version - before_block.version == expected_version_delta


@pytest.mark.asyncio
async def test_same_identity_shares_memory_across_sessions() -> None:
    trace_store, runtime = _runtime()
    service = AgentHarnessService.build_default(
        _settings(),
        model_gateway=StubModelGateway(),
        trace_store=trace_store,
        memory_runtime=runtime,
    )
    first_run = await service.runtime.start_run(
        AgentRequestContext(
            "tenant",
            "user",
            "agent",
            "session-a",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    await service.runtime.assemble_context(first_run, "保存长期规则")
    result = await service.graph.tool_gateway.execute(
        first_run,
        ModelToolCall(
            id="core-write",
            name="updateCoreMemory",
            arguments={
                "blockKey": "user_rules",
                "newContent": "所有回答先给证据。",
            },
        ),
    )
    second_run = await service.runtime.start_run(
        AgentRequestContext("tenant", "user", "agent", "session-b")
    )
    assembled = await service.runtime.assemble_context(second_run, "继续分析")

    assert result.result["success"] is True
    assert any("所有回答先给证据。" in message.content for message in assembled.messages)


@pytest.mark.asyncio
async def test_core_update_requires_snapshot_advances_version_and_cleans_up() -> None:
    trace_store, runtime = _runtime()
    service = AgentHarnessService.build_default(
        _settings(),
        model_gateway=StubModelGateway(),
        trace_store=trace_store,
        memory_runtime=runtime,
    )
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tenant",
            "user",
            "agent",
            "snapshot-required",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    def call(call_id: str, content: str) -> ModelToolCall:
        return ModelToolCall(
            id=call_id,
            name="updateCoreMemory",
            arguments={"blockKey": "user_rules", "newContent": content},
        )

    missing = await service.graph.tool_gateway.execute(
        run_context,
        call("missing", "先给证据。"),
    )
    await service.runtime.assemble_context(run_context, "更新长期规则")
    first = await service.graph.tool_gateway.execute(
        run_context,
        call("first", "先给证据。"),
    )
    second = await service.graph.tool_gateway.execute(
        run_context,
        call("second", "先给证据，再给结论。"),
    )

    assert missing.result["error_type"] == "update_context_missing"
    assert first.result["version"] == 2
    assert second.result["version"] == 3
    assert runtime.core_version_snapshots.contains(run_context.run_id)
    await service.runtime.complete_run(run_context, "done")
    assert not runtime.core_version_snapshots.contains(run_context.run_id)


@pytest.mark.asyncio
async def test_fail_run_and_duplicate_cleanup_release_core_snapshot() -> None:
    trace_store, runtime = _runtime()
    service = AgentHarnessService.build_default(
        _settings(),
        model_gateway=StubModelGateway(),
        trace_store=trace_store,
        memory_runtime=runtime,
    )
    run_context = await service.runtime.start_run(
        AgentRequestContext("tenant", "user", "agent", "failed-run")
    )
    await service.runtime.assemble_context(run_context, "读取 Core Memory")
    assert runtime.core_version_snapshots.contains(run_context.run_id)

    await service.runtime.fail_run(run_context, "failed")
    assert not runtime.core_version_snapshots.contains(run_context.run_id)
    service.runtime.cleanup_run(run_context.run_id)
    service.runtime.cleanup_run(run_context.run_id)
    assert not runtime.core_version_snapshots.contains(run_context.run_id)


@pytest.mark.asyncio
async def test_context_assembly_error_releases_snapshot_without_test_cleanup() -> None:
    trace_store, runtime = _runtime()
    service = AgentHarnessService.build_default(
        _settings(),
        model_gateway=StubModelGateway(),
        trace_store=trace_store,
        memory_runtime=runtime,
    )
    provider = service.runtime.context_manager.memory_context_provider
    assert provider is not None

    class FailAfterCapture:
        async def build_context(self, run_context):
            await provider.build_context(run_context)
            raise RuntimeError("context failed")

    service.runtime.context_manager.memory_context_provider = FailAfterCapture()
    result = await service.chat(
        AgentRequestContext("tenant", "user", "agent", "context-error"),
        "触发上下文异常",
    )

    assert result.success is False
    assert result.run_id is not None
    assert not runtime.core_version_snapshots.contains(result.run_id)


@pytest.mark.asyncio
async def test_snapshot_identity_mismatch_releases_snapshot_without_test_cleanup() -> None:
    trace_store, runtime = _runtime()
    service = AgentHarnessService.build_default(
        _settings(),
        model_gateway=StubModelGateway(),
        trace_store=trace_store,
        memory_runtime=runtime,
    )
    run_context = await service.runtime.start_run(
        AgentRequestContext("tenant", "user", "agent", "identity")
    )
    await service.runtime.assemble_context(run_context, "读取 Core Memory")
    mismatched = RunContext(
        request_context=AgentRequestContext("other", "user", "agent", "identity"),
        run_id=run_context.run_id,
        prompt_version=run_context.prompt_version,
        tool_schema_version=run_context.tool_schema_version,
        model_provider=run_context.model_provider,
    )

    with pytest.raises(MemoryStoreIsolationError):
        await service.runtime.assemble_context(mismatched, "错误身份")

    assert not runtime.core_version_snapshots.contains(run_context.run_id)


@pytest.mark.asyncio
async def test_in_memory_default_core_initialization_is_all_or_nothing() -> None:
    _trace_store, runtime = _runtime()
    admin = runtime.fixture_admin
    assert admin is not None
    archived = CoreMemoryBlock(
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        block_key="service_notes",
        description=CORE_BLOCK_SPECS["service_notes"].description,
        max_tokens=CORE_BLOCK_SPECS["service_notes"].max_tokens,
        status="archived",
    )
    await admin.upsert_core_block(archived)

    with pytest.raises(CoreMemoryContractError):
        await runtime.repository.ensure_default_core_blocks("tenant", "user", "agent")

    blocks = await runtime.inspection_repository.list_core_blocks_for_scope(
        "tenant", "user", "agent"
    )
    assert [(block.block_key, block.status) for block in blocks] == [
        ("service_notes", "archived")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"id": ""},
        {"tenant_id": "other"},
        {"user_id": "other"},
        {"agent_id": "other"},
        {"block_key": "other"},
        {"status": "archived"},
        {"version": 0},
        {"version": True},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"content": 7},
        {"content_hash": "wrong"},
    ],
)
async def test_in_memory_default_core_rejects_malformed_existing_before_writes(
    overrides: dict[str, object],
) -> None:
    _trace_store, runtime = _runtime()
    admin = runtime.fixture_admin
    assert isinstance(admin, InMemoryMemoryRepository)
    malformed = CoreMemoryBlock(
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        block_key="service_notes",
        description=CORE_BLOCK_SPECS["service_notes"].description,
        max_tokens=CORE_BLOCK_SPECS["service_notes"].max_tokens,
    )
    for field_name, value in overrides.items():
        setattr(malformed, field_name, value)
    admin.store._core_blocks[("tenant", "user", "agent", "service_notes")] = malformed

    with pytest.raises(CoreMemoryContractError):
        await runtime.repository.ensure_default_core_blocks("tenant", "user", "agent")

    blocks = await runtime.inspection_repository.list_core_blocks_for_scope(
        "tenant", "user", "agent"
    )
    assert len(blocks) == 1


@pytest.mark.asyncio
async def test_core_snapshot_capture_once_rejects_concurrent_replacement() -> None:
    trace_store, runtime = _runtime()
    service = AgentHarnessService.build_default(
        _settings(),
        model_gateway=StubModelGateway(),
        trace_store=trace_store,
        memory_runtime=runtime,
    )
    run_context = await service.runtime.start_run(
        AgentRequestContext(
            "tenant",
            "user",
            "agent",
            "capture-once",
            permissions=LOCAL_DEFAULT_PERMISSIONS,
            auth_mode="dev_headers",
        )
    )
    await service.runtime.assemble_context(run_context, "读取 Core Memory")
    admin = runtime.fixture_admin
    assert admin is not None
    current = next(
        block
        for block in await runtime.inspection_repository.list_core_blocks_for_scope(
            "tenant", "user", "agent"
        )
        if block.block_key == "user_rules"
    )
    await admin.upsert_core_block(
        replace(
            current,
            content="并发更新",
            content_hash=content_hash("并发更新"),
            version=current.version + 1,
        )
    )
    await service.runtime.assemble_context(run_context, "再次组装上下文")

    conflict = await service.graph.tool_gateway.execute(
        run_context,
        ModelToolCall(
            id="conflict",
            name="updateCoreMemory",
            arguments={"blockKey": "user_rules", "newContent": "当前 run 的更新"},
        ),
    )

    assert conflict.result["success"] is False
    assert conflict.result["error_type"] == "update_conflict"


@pytest.mark.asyncio
async def test_snapshots_are_isolated_by_tenant_user_and_agent() -> None:
    _trace_store, runtime = _runtime()
    seeder = MemoryFixtureSeeder(runtime)
    provider = MemorySnapshotProvider(runtime.inspection_repository)
    identities = [
        ("tenant-a", "user-a", "agent-a", "tenant-canary"),
        ("tenant-b", "user-a", "agent-a", "other-tenant-canary"),
        ("tenant-a", "user-b", "agent-a", "other-user-canary"),
        ("tenant-a", "user-a", "agent-b", "other-agent-canary"),
    ]
    for tenant_id, user_id, agent_id, canary in identities:
        await seeder.seed(
            tenant_id,
            user_id,
            agent_id,
            core_blocks={"user_rules": canary},
            archival_memories=[
                ArchivalMemoryFixture(
                    fixture_id=f"fixture-{canary}",
                    topic=canary,
                    content=f"archival-{canary}",
                )
            ],
        )

    primary = await provider.capture("tenant-a", "user-a", "agent-a")
    serialized = repr(primary)

    assert "tenant-canary" in serialized
    assert "other-tenant-canary" not in serialized
    assert "other-user-canary" not in serialized
    assert "other-agent-canary" not in serialized


@pytest.mark.asyncio
async def test_each_case_and_repetition_gets_a_fresh_memory_runtime() -> None:
    trace_a, runtime_a = _runtime()
    trace_b, runtime_b = _runtime()
    await MemoryFixtureSeeder(runtime_a).seed(
        "tenant",
        "user",
        "agent",
        core_blocks={"user_rules": "case-a-only"},
    )

    assert trace_a is not trace_b
    assert runtime_a.repository is not runtime_b.repository
    assert "case-a-only" in repr(
        await MemorySnapshotProvider(runtime_a.inspection_repository).capture(
            "tenant", "user", "agent"
        )
    )
    assert (await MemorySnapshotProvider(runtime_b.inspection_repository).capture(
        "tenant", "user", "agent"
    )).core_blocks == ()


def test_build_default_supports_injection_and_preserves_default_path() -> None:
    settings = _settings()
    trace_store, memory_runtime = _runtime()
    model_gateway = StubModelGateway()

    injected = AgentHarnessService.build_default(
        settings,
        model_gateway=model_gateway,
        trace_store=trace_store,
        memory_runtime=memory_runtime,
    )
    default = AgentHarnessService.build_default(settings)

    assert injected.trace_store is trace_store
    assert injected.memory_runtime is memory_runtime
    assert injected.graph.model_gateway is model_gateway
    assert injected.runtime.context_assembler.memory_context_provider is (
        memory_runtime.context_provider
    )
    assert default.memory_runtime is not None
    assert default.memory_runtime.repository is not memory_runtime.repository
    assert {tool.name for tool in default.graph.tool_gateway.registry.list()} >= {
        "updateCoreMemory",
        "saveArchivalMemory",
        "searchMemory",
    }


def test_build_default_rejects_mismatched_memory_trace_store() -> None:
    _trace_store, memory_runtime = _runtime()

    with pytest.raises(ValueError, match="same trace_store"):
        AgentHarnessService.build_default(
            _settings(),
            trace_store=InMemoryRolloutEventStore(),
            memory_runtime=memory_runtime,
        )
