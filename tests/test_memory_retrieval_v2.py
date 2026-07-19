from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from superbiz_agent.evals.memory_retrieval_v2 import (
    V2Observation,
    V2ReturnedEvidence,
    analyze_memory_retrieval_v2_quality,
    calibrate_v2,
    load_memory_retrieval_v2_dataset,
    serialize_v2_artifact,
)
from superbiz_agent.memory.embedding import MemoryEmbeddingIdentity


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "memory_retrieval_v2.json"
MANIFEST = ROOT / "evals" / "datasets" / "memory_retrieval_v2.manifest.json"


def test_v2_dataset_has_frozen_dev_holdout_tracks_and_manifest() -> None:
    dataset, sha = load_memory_retrieval_v2_dataset(DATASET, MANIFEST)
    assert len(dataset.fixtures) == 60
    assert len(dataset.queries) == 60
    assert sum(q.split == "dev" for q in dataset.queries) == 48
    assert sum(q.split == "holdout" for q in dataset.queries) == 12
    assert sha == json.loads(MANIFEST.read_text())["dataset_sha256"]
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == sha


def test_v2_quality_report_is_explicitly_pending_acceptance() -> None:
    dataset, _ = load_memory_retrieval_v2_dataset(DATASET, MANIFEST)
    report = analyze_memory_retrieval_v2_quality(dataset)
    assert report["status"] == "pending independent dataset acceptance"
    assert report["quality_gate"] is True
    assert report["track_counts"] == {"semantic_ranking": 30, "no_match": 10, "identity_isolation": 8}
    assert report["cross_split_near_duplicates"] == []


def test_v2_calibration_uses_observed_breakpoints_and_never_selects_production() -> None:
    dataset, _ = load_memory_retrieval_v2_dataset(DATASET, MANIFEST)
    dev = [q for q in dataset.queries if q.split == "dev"]
    observations = []
    for index, case in enumerate(dev):
        evidence = case.relevant_evidence_ids[0] if case.relevant_evidence_ids else "mpr2-e-canary-tenant-pg"
        returned = () if case.track != "semantic_ranking" else (V2ReturnedEvidence(1, evidence, 0.71 + index / 1000),)
        observations.append(V2Observation(case.case_id, case.split, case.track, case.category, hashlib.sha256(case.query.encode()).hexdigest(), returned, tuple(case.relevant_evidence_ids), tuple(case.forbidden_evidence_ids), tuple(case.relevant_evidence_ids[:1]), (), 0, bool(returned), 1.0, returned[0].similarity if returned else None, None, None, bool(returned)))
    report = calibrate_v2(observations, dataset)
    assert report["status"] in {"calibration_frontier_available", "no_simple_gate_separates_positive_and_negative"}
    assert report["production_candidate"] is None
    thresholds = {point["top1_threshold"] for point in report["points"]}
    assert 0.0 not in thresholds or len(thresholds) > 1
    encoded = serialize_v2_artifact(report)
    assert b"production_candidate" in encoded
    assert all(case.query.encode() not in encoded for case in dev)


def test_v2_artifact_serialization_is_deterministic() -> None:
    value = {"b": 2, "a": [1, 3], "nested": {"z": True}}
    assert serialize_v2_artifact(value) == serialize_v2_artifact(value)
    assert serialize_v2_artifact(value).endswith(b"\n")


def test_v2_report_manifest_hashes_report_calibration_and_frontier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import run_memory_retrieval_v2_baseline as baseline

    monkeypatch.setattr(baseline, "OUT", tmp_path)
    report_path, manifest_path, calibration_path, frontier_path = baseline._write_artifacts(
        {"status": "completed", "observations": []},
        {"status": "calibration_frontier_available", "pareto_frontier": []},
        "a" * 64,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert manifest["calibration_sha256"] == hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    assert manifest["pareto_frontier_sha256"] == hashlib.sha256(frontier_path.read_bytes()).hexdigest()
    assert manifest["dataset_sha256"] == "a" * 64


@pytest.mark.asyncio
async def test_v2_runner_executes_dev_only_and_never_submits_holdout_queries(
    monkeypatch,
) -> None:
    from superbiz_agent.evals import memory_retrieval_v2 as module

    dataset, sha = load_memory_retrieval_v2_dataset(DATASET, MANIFEST)
    submitted: list[str] = []

    class _Search:
        def __init__(self, *_args, **_kwargs):
            pass

        async def search_memory(self, _tenant, _user, _agent, query, *_args, **_kwargs):
            submitted.append(query)
            return []

    monkeypatch.setattr(module, "MemorySearchService", _Search)
    embedding = SimpleNamespace(
        identity=MemoryEmbeddingIdentity(
            provider="dashscope-openai-compatible",
            model="text-embedding-v4",
            version="text-embedding-v4",
            dimension=1024,
        )
    )
    runner = module.MemoryRetrievalV2Runner(
        object(),
        embedding,
        dataset,
        sha,
        postgresql_version="16.14",
        pgvector_version="0.8.5",
    )
    observations, errors = await runner.execute_dev()
    assert not errors
    assert len(observations) == 48
    assert submitted == [query.query for query in dataset.queries if query.split == "dev"]
    assert not set(submitted) & {query.query for query in dataset.queries if query.split == "holdout"}
