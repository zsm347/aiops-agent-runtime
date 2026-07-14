from __future__ import annotations

import pytest

from sqlalchemy.ext.asyncio import AsyncEngine

from superbiz_agent.config import Settings
from superbiz_agent.harness.trace_store import InMemoryRolloutEventStore
from superbiz_agent.memory.runtime import build_memory_runtime
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository


def _memory_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


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
