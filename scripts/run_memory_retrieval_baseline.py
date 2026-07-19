from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from sqlalchemy import delete, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from superbiz_agent.config import Settings
from superbiz_agent.evals.memory_retrieval import (
    MemoryRetrievalRunner,
    load_memory_retrieval_dataset,
    write_memory_retrieval_report,
)
from superbiz_agent.memory.embedding import (
    REAL_MEMORY_EMBEDDING_PROVIDERS,
    build_memory_embedding_service,
)
from superbiz_agent.memory.errors import MemoryPersistenceError
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.models import LongTermMemory
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals" / "datasets" / "memory_retrieval_v1.json"
MANIFEST_PATH = ROOT / "evals" / "datasets" / "memory_retrieval_v1.manifest.json"
DATABASE_URL_ENV = "M_P1_TEST_DATABASE_URL"
CONFIRMATION_ENV = "M_P1_TEST_DATABASE_DESTRUCTIVE_CONFIRM"
CONFIRMATION_VALUE = "ERASE_M_P1_ISOLATED_TEST_DATABASE"
PENDING_EXIT_CODE = 3
DEDICATED_MARKERS = frozenset(
    {
        "superbiz-agent:m-p2-destructive-test-database:v1",
        "superbiz-agent:m-p1-retrieval-test-database:v1",
    }
)
FORBIDDEN_DATABASES = frozenset(
    {"postgres", "template0", "template1", "super_biz_agent", "superbiz_agent"}
)


def _validated_url(value: str, confirmation: str | None) -> tuple[str, str]:
    try:
        parsed = make_url(value)
    except (ArgumentError, TypeError, ValueError):
        raise RuntimeError("memory retrieval database URL is invalid") from None
    database = (parsed.database or "").strip()
    if (
        confirmation != CONFIRMATION_VALUE
        or parsed.get_backend_name() != "postgresql"
        or parsed.get_driver_name() != "asyncpg"
        or not parsed.host
        or not database
        or database.lower() in FORBIDDEN_DATABASES
        or ("m_p1_test" not in database.lower() and "m_p2_test" not in database.lower())
    ):
        raise RuntimeError("memory retrieval database is not dedicated-test-only")
    return value, database


async def _run() -> int:
    raw_url = os.environ.get(DATABASE_URL_ENV)
    if not raw_url:
        print(
            json.dumps(
                {
                    "status": "pending",
                    "reason": "M_P1_TEST_DATABASE_URL_missing",
                    "exit_code": PENDING_EXIT_CODE,
                },
                sort_keys=True,
            )
        )
        return PENDING_EXIT_CODE

    engine = None
    embedding_service = None
    fixture_ids: list[str] = []
    try:
        database_url, expected_database = _validated_url(
            raw_url,
            os.environ.get(CONFIRMATION_ENV),
        )
        settings = Settings(
            database_url=database_url,
            memory_enabled=True,
            memory_store_backend="postgres",
        )
        if (
            settings.memory_embedding_provider not in REAL_MEMORY_EMBEDDING_PROVIDERS
            or settings.memory_embedding_dimension != 1024
            or not settings.memory_embedding_api_key
            or (
                settings.memory_embedding_provider == "dashscope-openai-compatible"
                and not settings.memory_embedding_base_url
            )
        ):
            print(
                json.dumps(
                    {
                        "status": "pending",
                        "reason": "MEMORY_EMBEDDING_configuration_missing",
                        "exit_code": PENDING_EXIT_CODE,
                    },
                    sort_keys=True,
                )
            )
            return PENDING_EXIT_CODE
        dataset, dataset_sha = load_memory_retrieval_dataset(DATASET_PATH, MANIFEST_PATH)
        fixture_ids = [fixture.fixture_id for fixture in dataset.fixtures]
        engine = create_engine(database_url)
        repository = PostgresMemoryRepository(create_sessionmaker(engine))
        embedding_service = build_memory_embedding_service(settings)
        await repository.ensure_ready()

        async with engine.connect() as connection:
            actual_database = await connection.scalar(text("SELECT current_database()"))
            marker = await connection.scalar(
                text(
                    "SELECT shobj_description(oid, 'pg_database') FROM pg_database "
                    "WHERE datname = current_database()"
                )
            )
            application_rows = await connection.scalar(
                text(
                    "SELECT (SELECT count(*) FROM agent_rollout_event) + "
                    "(SELECT count(*) FROM long_term_memory) + "
                    "(SELECT count(*) FROM agent_core_memory_block) + "
                    "(SELECT count(*) FROM rag_knowledge_base) + "
                    "(SELECT count(*) FROM rag_document)"
                )
            )
            postgresql_version = str(await connection.scalar(text("SHOW server_version")))
            pgvector_version = str(
                await connection.scalar(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
            )
        if (
            actual_database != expected_database
            or marker not in DEDICATED_MARKERS
            or int(application_rows or 0) != 0
        ):
            raise RuntimeError("memory retrieval database is not empty and dedicated")

        runner = MemoryRetrievalRunner(
            repository,
            embedding_service,
            dataset,
            dataset_sha,
            postgresql_version=postgresql_version,
            pgvector_version=pgvector_version,
            production_default_top_k=settings.memory_search_top_k,
            production_default_min_similarity=settings.memory_search_min_similarity,
        )
        await runner.seed_fixtures()
        report = await runner.run()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = (
            ROOT
            / "artifacts"
            / "evals"
            / "memory_retrieval"
            / f"{timestamp}-{dataset.version}-dev.json"
        )
        artifact_manifest, artifact_manifest_path = write_memory_retrieval_report(
            report,
            report_path,
        )
        candidate = report.candidate_evaluation
        print(
            json.dumps(
                {
                    "status": report.status,
                    "production_retrieval_ranking": report.production_retrieval_ranking,
                    "planned": report.planned_cases,
                    "executed": report.executed_cases,
                    "skipped": report.skipped_cases,
                    "infrastructure_failures": report.infrastructure_failures,
                    "scan_status": report.scan_status,
                    "production_candidate_status": report.production_candidate_status,
                    "selected_top_k": report.selected_top_k,
                    "selected_min_similarity": report.selected_min_similarity,
                    "quality_gate_passed": (
                        candidate.quality_gate.passed if candidate is not None else False
                    ),
                    "candidate_actual_p50_latency_ms": (
                        candidate.actual_query_p50_latency_ms if candidate is not None else None
                    ),
                    "candidate_actual_p95_latency_ms": (
                        candidate.actual_query_p95_latency_ms if candidate is not None else None
                    ),
                    "report_path": str(report_path.relative_to(ROOT)),
                    "report_sha256": artifact_manifest.report_sha256,
                    "report_manifest_path": str(artifact_manifest_path.relative_to(ROOT)),
                },
                sort_keys=True,
            )
        )
        return _report_exit_code(report)
    except MemoryPersistenceError as exc:
        print(json.dumps({"status": "failed", "error_codes": [exc.code]}, sort_keys=True))
        return 1
    except Exception:
        print(
            json.dumps(
                {"status": "failed", "error_codes": ["memory_retrieval_gate_error"]},
                sort_keys=True,
            )
        )
        return 1
    finally:
        if engine is not None and fixture_ids:
            try:
                async with engine.begin() as connection:
                    await connection.execute(
                        delete(LongTermMemory).where(LongTermMemory.id.in_(fixture_ids))
                    )
            except Exception:
                pass
        if embedding_service is not None:
            try:
                await embedding_service.aclose()
            except Exception:
                pass
        if engine is not None:
            await engine.dispose()


def main() -> int:
    return asyncio.run(_run())


def _report_exit_code(report) -> int:
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
