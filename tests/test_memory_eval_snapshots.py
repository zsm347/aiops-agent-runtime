from __future__ import annotations

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.evals.memory_fixtures import (
    ArchivalMemoryFixture,
    CoreMemoryFixture,
    MemoryFixtureSeeder,
)
from superbiz_agent.evals.memory_snapshots import MemorySnapshotProvider
from superbiz_agent.harness.context import AgentRequestContext
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
    provider = MemorySnapshotProvider(runtime.store)

    first = provider.capture("tenant", "user", "agent")
    second = provider.capture("tenant", "user", "agent")

    assert first == second
    assert first.core_blocks == ()
    assert first.archival_memories == ()
    assert runtime.store.list_core_blocks_for_scope("tenant", "user", "agent") == []
    assert await trace_store.list_by_session(
        AgentRequestContext("tenant", "user", "agent", "snapshot")
    ) == []


@pytest.mark.asyncio
async def test_fixture_initialization_precedes_snapshot_and_is_not_an_agent_write() -> None:
    trace_store, runtime = _runtime()
    seeder = MemoryFixtureSeeder(runtime)
    provider = MemorySnapshotProvider(runtime.store)

    seeded = seeder.seed(
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
    before = provider.capture("tenant", "user", "agent")

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
    provider = MemorySnapshotProvider(runtime.store)
    seeder.seed(
        "tenant",
        "user",
        "agent",
        core_blocks={
            block_key: CoreMemoryFixture(content=initial_content, version=initial_version)
        },
    )
    before = provider.capture("tenant", "user", "agent")
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
    result = await service.graph.tool_gateway.execute(
        run_context,
        ModelToolCall(
            id=f"update-{block_key}",
            name="updateCoreMemory",
            arguments={"blockKey": block_key, "newContent": next_content},
        ),
    )
    after = provider.capture("tenant", "user", "agent")
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


def test_snapshots_are_isolated_by_tenant_user_and_agent() -> None:
    _trace_store, runtime = _runtime()
    seeder = MemoryFixtureSeeder(runtime)
    provider = MemorySnapshotProvider(runtime.store)
    identities = [
        ("tenant-a", "user-a", "agent-a", "tenant-canary"),
        ("tenant-b", "user-a", "agent-a", "other-tenant-canary"),
        ("tenant-a", "user-b", "agent-a", "other-user-canary"),
        ("tenant-a", "user-a", "agent-b", "other-agent-canary"),
    ]
    for tenant_id, user_id, agent_id, canary in identities:
        seeder.seed(
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

    primary = provider.capture("tenant-a", "user-a", "agent-a")
    serialized = repr(primary)

    assert "tenant-canary" in serialized
    assert "other-tenant-canary" not in serialized
    assert "other-user-canary" not in serialized
    assert "other-agent-canary" not in serialized


def test_each_case_and_repetition_gets_a_fresh_memory_runtime() -> None:
    trace_a, runtime_a = _runtime()
    trace_b, runtime_b = _runtime()
    MemoryFixtureSeeder(runtime_a).seed(
        "tenant",
        "user",
        "agent",
        core_blocks={"user_rules": "case-a-only"},
    )

    assert trace_a is not trace_b
    assert runtime_a.store is not runtime_b.store
    assert "case-a-only" in repr(
        MemorySnapshotProvider(runtime_a.store).capture("tenant", "user", "agent")
    )
    assert MemorySnapshotProvider(runtime_b.store).capture(
        "tenant", "user", "agent"
    ).core_blocks == ()


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
    assert default.memory_runtime.store is not memory_runtime.store
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
