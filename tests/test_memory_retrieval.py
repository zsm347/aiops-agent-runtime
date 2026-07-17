from __future__ import annotations

import json
from pathlib import Path

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.evals.memory_retrieval import (
    MemoryRetrievalRunner,
    RetrievalObservation,
    load_memory_retrieval_dataset,
)
from superbiz_agent.memory.embedding import (
    MemoryEmbeddingBatch,
    MemoryEmbeddingIdentity,
    MemoryQueryEmbedding,
)
from superbiz_agent.memory.ports import ExactMemoryWriteResult


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


def _runner(repository=None):
    dataset, dataset_sha = load_memory_retrieval_dataset(DATASET, MANIFEST)
    return MemoryRetrievalRunner(
        repository or _WriteRepository(),
        _EmbeddingService(),
        dataset,
        dataset_sha,
        postgresql_version="16.14",
        pgvector_version="0.8.5",
    )


def test_retrieval_dataset_and_manifest_freeze_stable_evidence_catalog() -> None:
    dataset, dataset_sha = load_memory_retrieval_dataset(DATASET, MANIFEST)

    assert dataset.version == "1.0.0"
    assert dataset.split == "dev"
    assert len(dataset.fixtures) == 8
    assert len(dataset.cases) == 12
    assert len({fixture.fixture_id for fixture in dataset.fixtures}) == 8
    assert len({fixture.evidence_id for fixture in dataset.fixtures}) == 8
    assert dataset_sha == json.loads(MANIFEST.read_text(encoding="utf-8"))["dataset_sha256"]
    assert {case.category for case in dataset.cases} == {
        "positive",
        "paraphrase",
        "hard_negative",
        "no_match",
        "scope_filter",
        "identity_isolation",
    }


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
    assert all(memory.embedding_model == IDENTITY.model for memory in repository.memories)


def test_scan_metrics_keep_no_match_isolation_and_ranking_separate() -> None:
    runner = _runner()
    observations = []
    for case in runner.dataset.cases:
        if case.case_id == "MPR08":
            evidence = ("mpr1-evidence-orders-pool",)
            similarities = (0.35,)
        elif case.case_id == "MPR10":
            evidence = ("mpr1-evidence-other-tenant", case.relevant_evidence_ids[0])
            similarities = (0.9, 0.8)
        elif case.relevant_evidence_ids:
            evidence = (case.relevant_evidence_ids[0],)
            similarities = (0.8,)
        else:
            evidence = ()
            similarities = ()
        observations.append(
            RetrievalObservation(
                case_id=case.case_id,
                category=case.category,
                evidence_ids=evidence,
                similarities=similarities,
                latency_ms=float(len(observations) + 1),
            )
        )

    permissive = runner._scan_point(observations, top_k=3, min_similarity=0.0)
    strict = runner._scan_point(observations, top_k=3, min_similarity=0.5)

    assert permissive.no_match_false_positive_rate == 0.5
    assert strict.no_match_false_positive_rate == 0.0
    assert permissive.identity_isolation_violations == 1
    assert permissive.forbidden_result_violations == 1
    assert strict.hit_rate_at_3 == 1.0
    assert strict.recall_at_3 == 0.95
    assert 0.0 < strict.mrr < 1.0
    assert strict.p50_latency_ms == 6.0
    assert strict.p95_latency_ms == 12.0

    selected = runner._select_dev_point((permissive, strict))
    assert selected == strict
    assert runner._quality_errors(observations, strict) == [
        {
            "case_id": "MPR05",
            "category": "scope_filter",
            "error_code": "relevant_evidence_partial",
        },
        {
            "case_id": "MPR10",
            "category": "identity_isolation",
            "error_code": "forbidden_evidence_returned",
        }
    ]


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
