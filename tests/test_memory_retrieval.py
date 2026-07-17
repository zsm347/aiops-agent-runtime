from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.evals.memory_retrieval import (
    ISOLATION_CASE_IDS,
    NO_MATCH_CASE_IDS,
    SEMANTIC_CASE_IDS,
    CaseObservation,
    MemoryRetrievalRunner,
    ReturnedEvidence,
    load_memory_retrieval_dataset,
    serialize_memory_retrieval_report,
    write_memory_retrieval_report,
)
from superbiz_agent.memory.embedding import (
    MemoryEmbeddingBatch,
    MemoryEmbeddingIdentity,
    MemoryQueryEmbedding,
)
from superbiz_agent.memory.ports import ExactMemoryWriteResult
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "memory_retrieval_v1.json"
MANIFEST = ROOT / "evals" / "datasets" / "memory_retrieval_v1.manifest.json"
IDENTITY = MemoryEmbeddingIdentity(
    provider="dashscope-openai-compatible",
    model="text-embedding-v4",
    version="text-embedding-v4",
    dimension=1024,
)


class _EmbeddingService:
    identity = IDENTITY

    async def embed_documents(self, texts):
        values = tuple(texts)
        return MemoryEmbeddingBatch(
            vectors=tuple(tuple([float(index + 1)] * 1024) for index, _ in enumerate(values)),
            identity=self.identity,
        )

    async def embed_query(self, text: str) -> MemoryQueryEmbedding:
        return MemoryQueryEmbedding(vector=tuple([1.0] * 1024), identity=self.identity)

    async def aclose(self) -> None:
        return None


class _WriteRepository:
    def __init__(self) -> None:
        self.memories = []

    async def write_archival_exact(self, memory):
        self.memories.append(memory)
        return ExactMemoryWriteResult("written", memory, False)


class _PostgresRepository(PostgresMemoryRepository):
    def __init__(self) -> None:
        pass


def _runner(repository=None) -> MemoryRetrievalRunner:
    dataset, dataset_sha = load_memory_retrieval_dataset(DATASET, MANIFEST)
    return MemoryRetrievalRunner(
        repository or _WriteRepository(),
        _EmbeddingService(),
        dataset,
        dataset_sha,
        postgresql_version="16.14",
        pgvector_version="0.8.5",
        production_default_top_k=3,
        production_default_min_similarity=0.5,
    )


def _superset_observations(runner: MemoryRetrievalRunner) -> list[CaseObservation]:
    observations: list[CaseObservation] = []
    weak_primary = {
        "MPR10": "mpr1-evidence-orders-pool",
        "MPR11": "mpr1-evidence-payments-timeout",
        "MPR12": "mpr1-evidence-auth-rotation",
    }
    for index, case in enumerate(runner.dataset.cases, start=1):
        returned: list[ReturnedEvidence] = []
        if case.track == "semantic_ranking":
            returned.extend(
                ReturnedEvidence(rank=rank, evidence_id=evidence_id, similarity=0.9 - rank / 100)
                for rank, evidence_id in enumerate(case.relevant_evidence_ids, start=1)
            )
            if case.case_id == "MPR03":
                returned.append(
                    ReturnedEvidence(
                        rank=len(returned) + 1,
                        evidence_id=case.forbidden_evidence_ids[0],
                        similarity=0.55,
                    )
                )
        elif case.track == "no_match":
            returned.append(
                ReturnedEvidence(
                    rank=1,
                    evidence_id="mpr1-evidence-orders-pool",
                    similarity=0.35,
                )
            )
        else:
            returned.append(
                ReturnedEvidence(
                    rank=1,
                    evidence_id=weak_primary[case.case_id],
                    similarity=0.45,
                )
            )
        observations.append(
            runner._case_observation(
                case,
                tuple(returned),
                top_k=5,
                min_similarity=-1.0,
                latency_ms=float(index),
            )
        )
    return observations


def _candidate_observations(
    runner: MemoryRetrievalRunner,
    superset: list[CaseObservation],
) -> list[CaseObservation]:
    derived = runner._derive_observations(superset, top_k=3, min_similarity=0.6)
    return [
        replace(observation, query_latency_ms=float(index * 10))
        for index, observation in enumerate(derived, start=1)
    ]


async def _candidate_report():
    runner = _runner(_PostgresRepository())
    superset = _superset_observations(runner)
    candidate = _candidate_observations(runner, superset)
    runner._execute_cases = AsyncMock(  # type: ignore[method-assign]
        side_effect=[(superset, []), (candidate, [])]
    )
    report = await runner.run(
        top_k_values=(3,),
        min_similarity_values=(0.4, 0.6),
    )
    return runner, report


def test_retrieval_dataset_manifest_and_track_semantics_are_frozen() -> None:
    dataset, dataset_sha = load_memory_retrieval_dataset(DATASET, MANIFEST)

    assert dataset.version == "1.0.1"
    assert dataset.split == "dev"
    assert len(dataset.fixtures) == 8
    assert len(dataset.cases) == 12
    assert dataset_sha == "007b2c17505784f53ab8937f19d243949a7899a7d6da256de62639d147ac3cbe"
    assert dataset_sha == json.loads(MANIFEST.read_text(encoding="utf-8"))["dataset_sha256"]
    assert {case.case_id for case in dataset.cases if case.track == "semantic_ranking"} == (
        SEMANTIC_CASE_IDS
    )
    assert {case.case_id for case in dataset.cases if case.track == "no_match"} == (
        NO_MATCH_CASE_IDS
    )
    isolation = [case for case in dataset.cases if case.track == "identity_isolation"]
    assert {case.case_id for case in isolation} == ISOLATION_CASE_IDS
    assert all(
        case.expected_empty
        and not case.relevant_evidence_ids
        and len(case.forbidden_evidence_ids) == 1
        for case in isolation
    )


def test_retrieval_manifest_rejects_dataset_drift(tmp_path: Path) -> None:
    drifted = tmp_path / DATASET.name
    drifted.write_bytes(DATASET.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest"):
        load_memory_retrieval_dataset(drifted, MANIFEST)


@pytest.mark.asyncio
async def test_runner_seeds_real_vectors_with_stable_fixture_ids() -> None:
    repository = _WriteRepository()
    runner = _runner(repository)

    await runner.seed_fixtures()

    assert len(repository.memories) == 8
    assert {memory.id for memory in repository.memories} == {
        fixture.fixture_id for fixture in runner.dataset.fixtures
    }
    assert all(len(memory.embedding) == 1024 for memory in repository.memories)
    assert all(memory.embedding_provider == IDENTITY.provider for memory in repository.memories)


def test_tracks_have_independent_denominators_and_hard_quality_gates() -> None:
    runner = _runner()
    observations = _superset_observations(runner)

    unsafe = runner._scan_point(
        observations,
        top_k=3,
        min_similarity=0.4,
    )
    safe = runner._scan_point(
        observations,
        top_k=3,
        min_similarity=0.6,
    )

    assert unsafe.semantic_ranking.planned_cases == 7
    assert unsafe.no_match.planned_cases == 2
    assert unsafe.identity_isolation.planned_cases == 3
    assert unsafe.semantic_ranking.hit_rate_at_3 == 1.0
    assert unsafe.semantic_ranking.recall_at_3 == 1.0
    assert unsafe.semantic_ranking.hard_negative_forbidden_violations == 1
    assert unsafe.identity_isolation.failed_cases == 3
    assert unsafe.quality_gate.passed is False
    assert safe.semantic_ranking.hit_rate_at_3 == 1.0
    assert safe.no_match.false_positive_rate == 0.0
    assert safe.identity_isolation.passed_cases == 3
    assert safe.quality_gate.passed is True
    assert runner._select_dev_point((unsafe, safe)) == safe


def test_no_quality_gate_point_means_no_candidate() -> None:
    runner = _runner()
    observations = _superset_observations(runner)
    failed = runner._scan_point(observations, top_k=3, min_similarity=0.4)

    assert failed.quality_gate.passed is False
    assert runner._select_dev_point((failed,)) is None


@pytest.mark.asyncio
async def test_candidate_is_reexecuted_with_actual_configuration_and_latency() -> None:
    runner, report = await _candidate_report()

    assert runner._execute_cases.await_args_list[0].kwargs == {  # type: ignore[attr-defined]
        "top_k": 3,
        "min_similarity": -1.0,
    }
    assert runner._execute_cases.await_args_list[1].kwargs == {  # type: ignore[attr-defined]
        "top_k": 3,
        "min_similarity": 0.6,
    }
    assert report.scan_status == "completed"
    assert report.production_candidate_status == "dev_pilot_candidate"
    assert (report.selected_top_k, report.selected_min_similarity) == (3, 0.6)
    assert report.production_default_min_similarity == 0.5
    assert report.production_default_changed is False
    assert report.candidate_evaluation is not None
    assert report.candidate_evaluation.actual_query_p50_latency_ms == 60.0
    assert report.candidate_evaluation.actual_query_p95_latency_ms == 120.0
    assert all(
        observation.applied_top_k == 3
        and observation.applied_min_similarity == 0.6
        for observation in report.candidate_evaluation.observations
    )
    scan_point = next(point for point in report.scan if point.min_similarity == 0.6)
    assert scan_point.superset_scan_p50_latency_ms == 6.0
    assert scan_point.superset_scan_p95_latency_ms == 12.0


@pytest.mark.asyncio
async def test_report_observations_are_redacted_structured_and_ranked() -> None:
    _runner_value, report = await _candidate_report()
    observation = report.superset_scan_observations[0]
    source_query = "orders service PostgreSQL connection pool exhaustion"

    assert observation.query_sha256 == hashlib.sha256(source_query.encode()).hexdigest()
    assert observation.returned[0].rank == 1
    assert observation.returned[0].evidence_id == "mpr1-evidence-orders-pool"
    assert observation.relevant_matches == ("mpr1-evidence-orders-pool",)
    payload = serialize_memory_retrieval_report(report).decode("ascii")
    assert source_query not in payload
    assert "api_key" not in payload.lower()
    assert "postgresql+asyncpg" not in payload.lower()
    assert "mpr1-tenant" not in payload


@pytest.mark.asyncio
async def test_report_serialization_and_manifest_hash_are_deterministic(tmp_path: Path) -> None:
    _runner_value, report = await _candidate_report()
    first_bytes = serialize_memory_retrieval_report(report)
    second_bytes = serialize_memory_retrieval_report(report)
    report_path = tmp_path / "report.json"

    manifest, manifest_path = write_memory_retrieval_report(report, report_path)

    assert first_bytes == second_bytes == report_path.read_bytes()
    assert manifest.report_sha256 == hashlib.sha256(first_bytes).hexdigest()
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest == manifest.to_dict()
    assert persisted_manifest["dataset_sha256"] == report.dataset_sha256


@pytest.mark.asyncio
async def test_baseline_script_reports_pending_without_independent_memory_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import run_memory_retrieval_baseline as baseline

    settings = Settings(
        _env_file=None,
        memory_embedding_provider="local-deterministic",
        memory_embedding_model="local-deterministic",
        memory_embedding_dimension=64,
        memory_embedding_api_key=None,
        model_api_key="chat-secret-must-not-be-used",
        rag_embedding_api_key="rag-secret-must-not-be-used",
    )
    monkeypatch.setenv(
        "M_P1_TEST_DATABASE_URL",
        "postgresql+asyncpg://runner@localhost:5432/isolated_m_p1_test",
    )
    monkeypatch.setenv(
        "M_P1_TEST_DATABASE_DESTRUCTIVE_CONFIRM",
        "ERASE_M_P1_ISOLATED_TEST_DATABASE",
    )
    monkeypatch.setattr(baseline, "Settings", lambda **_kwargs: settings)

    exit_code = await baseline._run()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == baseline.PENDING_EXIT_CODE
    assert output == {
        "status": "pending",
        "reason": "MEMORY_EMBEDDING_configuration_missing",
        "exit_code": baseline.PENDING_EXIT_CODE,
    }


def test_real_memory_settings_remain_independent_from_rag_credentials() -> None:
    settings = Settings(
        _env_file=None,
        memory_embedding_provider="dashscope-openai-compatible",
        memory_embedding_model="text-embedding-v4",
        memory_embedding_version="text-embedding-v4",
        memory_embedding_dimension=1024,
        memory_embedding_base_url="https://example.invalid/compatible-mode/v1",
        memory_embedding_api_key=None,
        rag_embedding_api_key="rag-only-secret",
    )

    assert settings.memory_embedding_api_key is None
