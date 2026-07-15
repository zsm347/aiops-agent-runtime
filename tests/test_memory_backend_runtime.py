from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine

from superbiz_agent.api.app import create_app
from superbiz_agent.config import Settings
from superbiz_agent.evals.memory_runner import (
    MemoryEvalRunner,
    _configured_tool_schema_fingerprint,
)
from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.harness.stores import PostgresRolloutEventStore
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore
from superbiz_agent.memory.errors import MemoryStoreContractError
from superbiz_agent.memory.runtime import build_memory_runtime
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository


def _memory_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class _ControlledRepository:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.calls = 0
        self.failure = failure
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def ensure_ready(self) -> None:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure


class _ControlledEngine:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.calls = 0
        self.failure = failure
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def dispose(self) -> None:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        failure = self.failure
        self.failure = None
        if failure is not None:
            raise failure


class _CloseableRag:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.calls = 0
        self.failure = failure

    async def aclose(self) -> None:
        self.calls += 1
        failure = self.failure
        self.failure = None
        if failure is not None:
            raise failure


class _CloseableTraceStore(InMemoryRolloutEventStore):
    def __init__(self, failure: BaseException | None = None) -> None:
        super().__init__()
        self.calls = 0
        self.failure = failure

    async def aclose(self) -> None:
        self.calls += 1
        failure = self.failure
        self.failure = None
        if failure is not None:
            raise failure


def _controlled_postgres_runtime(repository, engine, *, trace_store=None):
    trace_store = trace_store or InMemoryRolloutEventStore()
    runtime = build_memory_runtime(
        _memory_settings(memory_store_backend="memory"),
        trace_store=trace_store,
    )
    runtime.backend = "postgres"
    runtime.repository = repository
    runtime.inspection_repository = repository
    runtime.fixture_admin = None
    runtime.engine = engine
    return trace_store, runtime


def test_invalid_memory_store_backend_fails_settings_validation() -> None:
    with pytest.raises(ValueError):
        _memory_settings(memory_store_backend="sqlite")


def test_postgres_backend_requires_memory_enabled() -> None:
    with pytest.raises(ValueError):
        _memory_settings(memory_store_backend="postgres", memory_enabled=False)


@pytest.mark.asyncio
async def test_memory_backend_runtime_is_isolated_and_ready() -> None:
    settings = _memory_settings(memory_store_backend="memory")
    trace_store = InMemoryRolloutEventStore()
    runtime = build_memory_runtime(settings, trace_store=trace_store)

    assert runtime.backend == "memory"
    assert runtime.engine is None
    assert runtime.fixture_admin is not None
    await runtime.ensure_ready()  # no-op for memory backend
    await runtime.aclose()  # no-op for memory backend


@pytest.mark.asyncio
async def test_postgres_backend_builds_engine_and_rejects_fixtures() -> None:
    settings = _memory_settings(
        memory_store_backend="postgres",
        database_url="postgresql+asyncpg://postgres:postgres@localhost:5432/super_biz_agent",
    )
    trace_store = InMemoryRolloutEventStore()
    runtime = build_memory_runtime(settings, trace_store=trace_store)

    assert runtime.backend == "postgres"
    assert isinstance(runtime.repository, PostgresMemoryRepository)
    assert isinstance(runtime.engine, AsyncEngine)
    assert runtime.fixture_admin is None
    assert runtime.inspection_repository is runtime.repository
    await runtime.aclose()


@pytest.mark.asyncio
async def test_postgres_backend_aclose_disposes_engine() -> None:
    settings = _memory_settings(
        memory_store_backend="postgres",
        database_url="postgresql+asyncpg://postgres:postgres@localhost:5432/super_biz_agent",
    )
    trace_store = InMemoryRolloutEventStore()
    runtime = build_memory_runtime(settings, trace_store=trace_store)
    engine = runtime.engine
    assert engine is not None

    await runtime.aclose()
    # Repeated close must be safe and idempotent (D3 lifecycle contract).
    await runtime.aclose()


@pytest.mark.asyncio
async def test_readiness_is_shared_run_once_and_survives_waiter_cancellation() -> None:
    repository = _ControlledRepository()
    engine = _ControlledEngine()
    engine.release.set()
    trace_store, runtime = _controlled_postgres_runtime(repository, engine)
    service = AgentHarnessService.build_default(
        _memory_settings(memory_store_backend="postgres"),
        trace_store=trace_store,
        memory_runtime=runtime,
        rag_retrieval_service=_CloseableRag(),  # type: ignore[arg-type]
    )

    cancelled_waiter = asyncio.create_task(service.ensure_ready())
    shared_waiter = asyncio.create_task(service.ensure_ready())
    await repository.started.wait()
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    repository.release.set()
    await shared_waiter
    await service.ensure_ready()

    assert repository.calls == 1


@pytest.mark.asyncio
async def test_readiness_failure_is_cached_and_remains_fail_closed() -> None:
    repository = _ControlledRepository(MemoryStoreContractError())
    repository.release.set()
    engine = _ControlledEngine()
    engine.release.set()
    trace_store, runtime = _controlled_postgres_runtime(repository, engine)
    service = AgentHarnessService.build_default(
        _memory_settings(memory_store_backend="postgres"),
        trace_store=trace_store,
        memory_runtime=runtime,
        rag_retrieval_service=_CloseableRag(),  # type: ignore[arg-type]
    )

    with pytest.raises(MemoryStoreContractError):
        await service.ensure_ready()
    with pytest.raises(MemoryStoreContractError):
        await service.ensure_ready()

    assert repository.calls == 1


@pytest.mark.asyncio
async def test_memory_runtime_close_waits_for_probe_and_rejects_late_readiness() -> None:
    repository = _ControlledRepository()
    engine = _ControlledEngine()
    engine.release.set()
    _trace_store, runtime = _controlled_postgres_runtime(repository, engine)
    readiness_waiter = asyncio.create_task(runtime.ensure_ready())
    await repository.started.wait()
    readiness_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await readiness_waiter

    close_task = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0)
    assert engine.calls == 0
    repository.release.set()
    await close_task

    assert repository.calls == 1
    assert engine.calls == 1
    with pytest.raises(RuntimeError, match="closing or closed"):
        await runtime.ensure_ready()


@pytest.mark.asyncio
async def test_harness_close_waits_for_startup_probe_and_rejects_chat_after_close() -> None:
    repository = _ControlledRepository()
    engine = _ControlledEngine()
    engine.release.set()
    trace_store, runtime = _controlled_postgres_runtime(repository, engine)
    service = AgentHarnessService.build_default(
        _memory_settings(memory_store_backend="postgres"),
        trace_store=trace_store,
        memory_runtime=runtime,
        rag_retrieval_service=_CloseableRag(),  # type: ignore[arg-type]
    )
    startup_waiter = asyncio.create_task(service.astart())
    await repository.started.wait()
    startup_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup_waiter

    close_task = asyncio.create_task(service.aclose())
    await asyncio.sleep(0)
    assert engine.calls == 0
    repository.release.set()
    await close_task

    with pytest.raises(RuntimeError, match="closing or closed"):
        await service.ensure_ready()
    result = await service.chat(
        AgentRequestContext("tenant", "user", "agent", "closed"),
        "must fail",
    )
    assert result.success is False
    assert result.error_message == "Harness service is closing or closed."


@pytest.mark.asyncio
async def test_memory_runtime_close_is_shared_cancellation_safe_and_retryable() -> None:
    repository = _ControlledRepository()
    repository.release.set()
    engine = _ControlledEngine()
    _trace_store, runtime = _controlled_postgres_runtime(repository, engine)
    cancelled_waiter = asyncio.create_task(runtime.aclose())
    shared_waiter = asyncio.create_task(runtime.aclose())
    await engine.started.wait()
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    engine.release.set()
    await shared_waiter
    await runtime.aclose()
    assert engine.calls == 1

    retry_engine = _ControlledEngine(RuntimeError("dispose failed"))
    retry_engine.release.set()
    _trace_store, retry_runtime = _controlled_postgres_runtime(repository, retry_engine)
    with pytest.raises(RuntimeError, match="dispose failed"):
        await retry_runtime.aclose()
    await retry_runtime.aclose()
    assert retry_engine.calls == 2


@pytest.mark.asyncio
async def test_harness_close_is_shared_and_survives_cancelled_waiter() -> None:
    repository = _ControlledRepository()
    repository.release.set()
    engine = _ControlledEngine()
    rag = _CloseableRag()
    trace_store = _CloseableTraceStore()
    _trace_store, runtime = _controlled_postgres_runtime(
        repository,
        engine,
        trace_store=trace_store,
    )
    service = AgentHarnessService.build_default(
        _memory_settings(memory_store_backend="postgres"),
        trace_store=trace_store,
        memory_runtime=runtime,
        rag_retrieval_service=rag,  # type: ignore[arg-type]
    )

    cancelled_waiter = asyncio.create_task(service.aclose())
    shared_waiter = asyncio.create_task(service.aclose())
    await engine.started.wait()
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    engine.release.set()
    await shared_waiter
    await service.aclose()

    assert (engine.calls, rag.calls, trace_store.calls) == (1, 1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_resource", ["memory", "rag", "rollout"])
async def test_harness_close_attempts_all_resources_and_retries_only_failure(
    failing_resource: str,
) -> None:
    repository = _ControlledRepository()
    repository.release.set()
    engine = _ControlledEngine(
        RuntimeError("memory failed") if failing_resource == "memory" else None
    )
    engine.release.set()
    rag = _CloseableRag(RuntimeError("rag failed") if failing_resource == "rag" else None)
    trace_store = _CloseableTraceStore(
        RuntimeError("rollout failed") if failing_resource == "rollout" else None
    )
    _trace_store, runtime = _controlled_postgres_runtime(
        repository,
        engine,
        trace_store=trace_store,
    )
    service = AgentHarnessService.build_default(
        _memory_settings(memory_store_backend="postgres"),
        trace_store=trace_store,
        memory_runtime=runtime,
        rag_retrieval_service=rag,  # type: ignore[arg-type]
    )

    expected_label = (
        "rollout_store"
        if failing_resource == "rollout"
        else f"{failing_resource}_runtime"
    )
    with pytest.raises(RuntimeError, match=expected_label):
        await service.aclose()
    assert (engine.calls, rag.calls, trace_store.calls) == (1, 1, 1)

    await service.aclose()
    await service.aclose()
    assert (engine.calls, rag.calls, trace_store.calls) == (
        (2, 1, 1)
        if failing_resource == "memory"
        else (1, 2, 1)
        if failing_resource == "rag"
        else (1, 1, 2)
    )


@pytest.mark.asyncio
async def test_postgres_rollout_store_close_is_shared_cancellation_safe_and_retryable() -> None:
    engine = _ControlledEngine()
    store = PostgresRolloutEventStore(object(), engine=engine)
    cancelled_waiter = asyncio.create_task(store.aclose())
    shared_waiter = asyncio.create_task(store.aclose())
    await engine.started.wait()
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    engine.release.set()
    await shared_waiter
    await store.aclose()
    assert engine.calls == 1
    with pytest.raises(RuntimeError, match="closing or closed"):
        await store.list_by_run("tenant", "run")

    retry_engine = _ControlledEngine(RuntimeError("rollout dispose failed"))
    retry_engine.release.set()
    retry_store = PostgresRolloutEventStore(object(), engine=retry_engine)
    with pytest.raises(RuntimeError, match="rollout dispose failed"):
        await retry_store.aclose()
    await retry_store.aclose()
    assert retry_engine.calls == 2


def test_postgres_rollout_store_builder_retains_owned_engine(monkeypatch) -> None:
    engine = object()
    sessionmaker = object()
    repository = object()
    monkeypatch.setattr(
        "superbiz_agent.persistence.database.create_engine",
        lambda _url: engine,
    )
    monkeypatch.setattr(
        "superbiz_agent.persistence.database.create_sessionmaker",
        lambda actual_engine: sessionmaker if actual_engine is engine else None,
    )
    monkeypatch.setattr(
        "superbiz_agent.persistence.repositories.rollout_events.RolloutEventRepository",
        lambda actual_sessionmaker: repository
        if actual_sessionmaker is sessionmaker
        else None,
    )

    store = AgentHarnessService._build_trace_store(
        _memory_settings(rollout_store_backend="postgres")
    )

    assert isinstance(store, PostgresRolloutEventStore)
    assert store.repository is repository
    assert store.engine is engine


def test_fastapi_startup_failure_closes_all_owned_resources() -> None:
    repository = _ControlledRepository(MemoryStoreContractError())
    repository.release.set()
    engine = _ControlledEngine()
    engine.release.set()
    rag = _CloseableRag()
    trace_store = _CloseableTraceStore()
    _trace_store, runtime = _controlled_postgres_runtime(
        repository,
        engine,
        trace_store=trace_store,
    )
    service = AgentHarnessService.build_default(
        _memory_settings(memory_store_backend="postgres"),
        trace_store=trace_store,
        memory_runtime=runtime,
        rag_retrieval_service=rag,  # type: ignore[arg-type]
    )
    app = create_app(
        _memory_settings(memory_enabled=False, memory_store_backend="memory")
    )
    app.state.harness_service = service

    with pytest.raises(MemoryStoreContractError) as exc_info:
        with TestClient(app):
            pass

    assert "schema contract" in str(exc_info.value)
    assert (repository.calls, engine.calls, rag.calls, trace_store.calls) == (1, 1, 1, 1)


def test_injected_memory_runtime_backend_must_match_settings() -> None:
    trace_store = InMemoryRolloutEventStore()
    runtime = build_memory_runtime(
        _memory_settings(memory_store_backend="memory"),
        trace_store=trace_store,
    )

    with pytest.raises(ValueError, match="does not match Settings"):
        AgentHarnessService.build_default(
            _memory_settings(memory_store_backend="postgres"),
            trace_store=trace_store,
            memory_runtime=runtime,
        )


def test_memory_eval_rejects_postgres_before_runtime_or_fingerprint_creation(
    monkeypatch,
) -> None:
    settings = _memory_settings(memory_store_backend="postgres")
    monkeypatch.setattr(
        "superbiz_agent.evals.memory_runner.build_memory_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("postgres engine must not be created")
        ),
    )

    with pytest.raises(ValueError, match="memory_store_backend='memory'"):
        MemoryEvalRunner(settings)
    with pytest.raises(ValueError, match="memory_store_backend='memory'"):
        _configured_tool_schema_fingerprint(settings)
