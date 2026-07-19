"""Run one redacted M-P1-R2 dev baseline in an isolated PostgreSQL database."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from sqlalchemy import delete, text
from sqlalchemy.engine import make_url

from superbiz_agent.config import Settings
from superbiz_agent.evals.memory_retrieval_v2 import (
    MemoryRetrievalV2Runner,
    calibrate_v2,
    load_memory_retrieval_v2_dataset,
    serialize_v2_artifact,
    summarize_v2_tracks,
)
from superbiz_agent.memory.embedding import REAL_MEMORY_EMBEDDING_PROVIDERS, build_memory_embedding_service
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.models import LongTermMemory
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "memory_retrieval_v2.json"
MANIFEST = ROOT / "evals" / "datasets" / "memory_retrieval_v2.manifest.json"
OUT = ROOT / "artifacts" / "evals" / "memory_retrieval_v2"
URL_ENV = "M_P1_R2_TEST_DATABASE_URL"
CONFIRM_ENV = "M_P1_R2_TEST_DATABASE_DESTRUCTIVE_CONFIRM"
CONFIRM = "ERASE_M_P1_R2_ISOLATED_TEST_DATABASE"
PENDING = 3
DATABASE_MARKER = "superbiz-agent:m-p1-r2-destructive-test-database:v1"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _dedicated_url(raw: str) -> str:
    parsed = make_url(raw)
    db = (parsed.database or "").lower()
    if parsed.get_backend_name() != "postgresql" or parsed.get_driver_name() != "asyncpg" or not parsed.host or "m_p1_r2_test" not in db or db in {"postgres", "template0", "template1", "super_biz_agent", "superbiz_agent"}:
        raise RuntimeError("database is not a dedicated M-P1-R2 test database")
    if os.environ.get(CONFIRM_ENV) != CONFIRM:
        raise RuntimeError("destructive confirmation is missing")
    return raw


def _write_artifacts(report: dict, calibration: dict, dataset_sha: str) -> tuple[Path, Path, Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = OUT / f"{stamp}-2.0.0-dev.json"
    report_bytes = serialize_v2_artifact(report)
    report_path.write_bytes(report_bytes)
    calibration_path = OUT / f"{stamp}-2.0.0-calibration.json"
    calibration_bytes = serialize_v2_artifact(calibration)
    calibration_path.write_bytes(calibration_bytes)
    frontier_path = OUT / f"{stamp}-2.0.0-pareto-frontier.json"
    frontier_bytes = serialize_v2_artifact({"status": calibration["status"], "dataset_sha256": dataset_sha, "pareto_frontier": calibration.get("pareto_frontier", [])})
    frontier_path.write_bytes(frontier_bytes)
    manifest = {"schema_version": "1.0.0", "report_file": report_path.name, "report_sha256": hashlib.sha256(report_bytes).hexdigest(), "calibration_file": calibration_path.name, "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(), "pareto_frontier_file": frontier_path.name, "pareto_frontier_sha256": hashlib.sha256(frontier_bytes).hexdigest(), "dataset_sha256": dataset_sha}
    manifest_path = report_path.with_name(f"{report_path.stem}.manifest.json")
    manifest_path.write_bytes(serialize_v2_artifact(manifest))
    return report_path, manifest_path, calibration_path, frontier_path


async def _run() -> int:
    raw_url = os.environ.get(URL_ENV)
    if not raw_url:
        print(json.dumps({"status": "pending", "reason": f"{URL_ENV}_missing", "exit_code": PENDING}, sort_keys=True))
        return PENDING
    required = ("MEMORY_EMBEDDING_PROVIDER", "MEMORY_EMBEDDING_MODEL", "MEMORY_EMBEDDING_VERSION", "MEMORY_EMBEDDING_DIMENSION", "MEMORY_EMBEDDING_API_KEY", "MEMORY_EMBEDDING_BASE_URL")
    if any(not os.environ.get(name) for name in required):
        print(json.dumps({"status": "pending", "reason": "MEMORY_EMBEDDING_configuration_missing", "exit_code": PENDING}, sort_keys=True))
        return PENDING
    engine = None
    embedding = None
    fixture_ids: tuple[str, ...] = ()
    try:
        database_url = _dedicated_url(raw_url)
        settings = Settings(_env_file=None, database_url=database_url, memory_enabled=True, memory_store_backend="postgres", memory_embedding_provider=os.environ["MEMORY_EMBEDDING_PROVIDER"], memory_embedding_model=os.environ["MEMORY_EMBEDDING_MODEL"], memory_embedding_version=os.environ["MEMORY_EMBEDDING_VERSION"], memory_embedding_dimension=int(os.environ["MEMORY_EMBEDDING_DIMENSION"]), memory_embedding_batch_size=10, memory_embedding_api_key=os.environ["MEMORY_EMBEDDING_API_KEY"], memory_embedding_base_url=os.environ["MEMORY_EMBEDDING_BASE_URL"])
        if settings.memory_embedding_provider not in REAL_MEMORY_EMBEDDING_PROVIDERS or settings.memory_embedding_dimension != 1024:
            raise RuntimeError("real 1024-dimension memory embedding is required")
        dataset, dataset_sha = load_memory_retrieval_v2_dataset(DATASET, MANIFEST)
        engine = create_engine(database_url)
        repository = PostgresMemoryRepository(create_sessionmaker(engine))
        embedding = build_memory_embedding_service(settings)
        await repository.ensure_ready()
        async with engine.connect() as connection:
            marker = await connection.scalar(text("SELECT shobj_description(oid, 'pg_database') FROM pg_database WHERE datname=current_database()"))
            owner = await connection.scalar(text("SELECT pg_get_userbyid(datdba) FROM pg_database WHERE datname=current_database()"))
            db_version = str(await connection.scalar(text("SHOW server_version")))
            vector_version = str(await connection.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'")))
            rows = await connection.scalar(text("SELECT (SELECT count(*) FROM agent_rollout_event) + (SELECT count(*) FROM long_term_memory) + (SELECT count(*) FROM agent_core_memory_block) + (SELECT count(*) FROM rag_knowledge_base) + (SELECT count(*) FROM rag_document)"))
        if marker != DATABASE_MARKER or owner != "m_p1_r2_test_role" or int(rows or 0) != 0 or vector_version != "0.8.5":
            raise RuntimeError("dedicated database is not empty")
        runner = MemoryRetrievalV2Runner(repository, embedding, dataset, dataset_sha, postgresql_version=db_version, pgvector_version=vector_version)
        fixture_ids = await runner.seed_fixtures()
        observations, errors = await runner.execute_dev()
        query_attempts = len(observations) + len(errors)
        latencies = [observation.query_latency_ms for observation in observations]
        report = {"schema_version": "2.0.0", "status": "completed" if not errors and len(observations) == 48 else "infrastructure_failed", "dataset_sha256": dataset_sha, "embedding": {"provider": embedding.identity.provider, "model": embedding.identity.model, "version": embedding.identity.version, "dimension": embedding.identity.dimension, "api_requests_planned": 54, "api_requests_calculated_from_service_contract": 6 + query_attempts}, "postgresql_version": db_version, "pgvector_version": vector_version, "planned": 48, "executed": len(observations), "skipped": 48 - len(observations), "infrastructure_failures": len(errors), "error_codes": sorted({code for _, code in errors}), "top_k": 10, "min_similarity": -1.0, "superset_scan_latency_ms": {"p50": _percentile(latencies, 0.50), "p95": _percentile(latencies, 0.95)}, "tracks": summarize_v2_tracks(observations), "observations": [o.to_dict() for o in observations], "cleanup": {"performed": False, "remaining_rows": None}}
        calibration = calibrate_v2(observations, dataset) if not errors else {"status": "infrastructure_pending", "production_candidate": None}
        report["calibration_status"] = calibration["status"]
        async with engine.begin() as connection:
            await connection.execute(delete(LongTermMemory).where(LongTermMemory.id.in_(fixture_ids)))
            remaining = await connection.scalar(text("SELECT count(*) FROM long_term_memory WHERE session_id='mpr2-retrieval-baseline'"))
        report["cleanup"] = {"performed": True, "remaining_rows": int(remaining or 0)}
        report_path, manifest_path, calibration_path, frontier_path = _write_artifacts(report, calibration, dataset_sha)
        print(json.dumps({"status": report["status"], "planned": 48, "executed": len(observations), "skipped": 48 - len(observations), "infrastructure_failures": len(errors), "dataset_sha256": dataset_sha, "report": str(report_path.relative_to(ROOT)), "manifest": str(manifest_path.relative_to(ROOT)), "calibration": str(calibration_path.relative_to(ROOT)), "pareto_frontier": str(frontier_path.relative_to(ROOT))}, sort_keys=True))
        return 0 if report["status"] == "completed" else 1
    except Exception:
        print(json.dumps({"status": "failed", "error_codes": ["memory_retrieval_v2_gate_error"]}, sort_keys=True))
        return 1
    finally:
        if engine is not None and fixture_ids:
            try:
                async with engine.begin() as connection:
                    await connection.execute(delete(LongTermMemory).where(LongTermMemory.id.in_(fixture_ids)))
            except Exception:
                pass
        if embedding is not None:
            try:
                await embedding.aclose()
            except Exception:
                pass
        if engine is not None:
            await engine.dispose()


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
