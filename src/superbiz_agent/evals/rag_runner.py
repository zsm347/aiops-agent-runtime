from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping
from uuid import uuid4

from llama_index.core.evaluation.retrieval.metrics import (
    AveragePrecision,
    HitRate,
    MRR,
    NDCG,
    Precision,
    Recall,
)
from pydantic import BaseModel, ConfigDict, Field

from superbiz_agent.evals.rag_cases import (
    RAG_F0_TENANT_ID,
    RagF0ContractError,
    RagF0Dataset,
    RagF0EvidenceMapper,
    RagF0Query,
)
from superbiz_agent.rag.models import RagRetrievalMode, RagRetrievalRequest, RagRetrievalScope
from superbiz_agent.rag.retrieval import RagRetrievalService


class RagF0ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RagF0MetricSummary(RagF0ReportModel):
    value: float | None
    ci95_low: float | None
    ci95_high: float | None
    query_count: int = Field(ge=0)


class RagF0RetrievedEvidence(RagF0ReportModel):
    rank: int
    evidence_id: str
    chunk_id: str
    document_name: str
    heading_path: tuple[str, ...]
    score: float | None


class RagF0QueryResult(RagF0ReportModel):
    query_id: str
    query: str
    slices: tuple[str, ...]
    allow_no_answer: bool
    latency_ms: float
    expected_evidence_ids: tuple[str, ...]
    retrieved: tuple[RagF0RetrievedEvidence, ...]
    ranked_evidence_ids: tuple[str, ...]
    metrics: dict[str, float]
    failure: str | None = None


class RagF0Report(RagF0ReportModel):
    status: Literal["completed", "infrastructure_pending"]
    acceptance: Literal["pending_independent_dataset_acceptance"]
    baseline: Literal[
        "dense_only_no_rerank",
        "bm25_only_no_rerank",
        "dense_native_bm25_rrf_k60_no_rerank",
    ]
    dataset_sha256: str
    dataset_version: str
    split: Literal["dev"]
    query_count: int
    execution: dict[str, int]
    infrastructure_failure_count: int
    embedding: dict[str, str | int]
    infrastructure_versions: dict[str, str]
    metrics: dict[str, RagF0MetricSummary]
    slice_metrics: dict[str, dict[str, RagF0MetricSummary]]
    latency_ms: dict[str, float | None]
    per_query: tuple[RagF0QueryResult, ...]
    retrieval_config: dict[str, Any]
    dependency_versions: dict[str, str]
    ablations: dict[str, str]
    out_of_scope: dict[str, str]
    started_at: str
    finished_at: str


class RagF0Runner:
    def __init__(
        self,
        *,
        retrieval_service: RagRetrievalService,
        dataset: RagF0Dataset,
        bootstrap_samples: int = 2000,
        bootstrap_seed: int = 20260717,
        embedding_identity: Mapping[str, str | int] | None = None,
        infrastructure_versions: Mapping[str, str] | None = None,
        retrieval_mode: RagRetrievalMode = RagRetrievalMode.HYBRID,
    ) -> None:
        if bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        self._retrieval = retrieval_service
        self._dataset = dataset
        self._mapper = RagF0EvidenceMapper(dataset)
        self._bootstrap_samples = bootstrap_samples
        self._bootstrap_seed = bootstrap_seed
        self._embedding_identity = dict(embedding_identity or {})
        self._infrastructure_versions = dict(infrastructure_versions or {})
        self._retrieval_mode = retrieval_mode

    async def run_dev(self) -> RagF0Report:
        started_at = _utc_now()
        results: list[RagF0QueryResult] = []
        infrastructure_failures = 0
        for query in self._dataset.queries_for_split("dev"):
            result = await self._run_query(query)
            results.append(result)
            infrastructure_failures += result.failure is not None

        if infrastructure_failures:
            metrics = {}
            slice_metrics = {}
            completed_latencies = []
        else:
            metrics = self._aggregate_metrics(results)
            slice_metrics = self._slice_metrics(results)
            completed_latencies = [row.latency_ms for row in results]
        return RagF0Report(
            status="completed" if infrastructure_failures == 0 else "infrastructure_pending",
            acceptance="pending_independent_dataset_acceptance",
            baseline=_baseline_name(self._retrieval_mode),
            dataset_sha256=self._dataset.manifest.dataset_sha256,
            dataset_version=self._dataset.manifest.dataset_version,
            split="dev",
            query_count=len(results),
            execution={
                "planned": len(results),
                "executed": len(results),
                "skipped": 0,
                "failed": infrastructure_failures,
            },
            infrastructure_failure_count=infrastructure_failures,
            embedding=self._embedding_identity,
            infrastructure_versions=self._infrastructure_versions,
            metrics=metrics,
            slice_metrics=slice_metrics,
            latency_ms={
                "p50": _percentile(completed_latencies, 0.50),
                "p95": _percentile(completed_latencies, 0.95),
            },
            per_query=tuple(results),
            retrieval_config={
                "hybrid_top_k": 10,
                "reported_top_k": 10,
                "default_final_top_k": 3,
                "rrf_k": 60,
                "rerank_enabled": False,
                "top3_source": "prefix_of_same_top10_ranking",
                "retrieval_mode": self._retrieval_mode.value,
            },
            dependency_versions={
                "llama-index-core": version("llama-index-core"),
                "llama-index-vector-stores-milvus": version("llama-index-vector-stores-milvus"),
                "pymilvus": version("pymilvus"),
                "milvus-lite": version("milvus-lite"),
                "openai": version("openai"),
            },
            ablations={
                mode.value: (
                    "executed" if mode is self._retrieval_mode and infrastructure_failures == 0
                    else "infrastructure_pending"
                    if mode is self._retrieval_mode
                    else "not_executed_in_this_run"
                )
                for mode in RagRetrievalMode
            },
            out_of_scope={
                "tenant_isolation": "out_of_scope: retained in Batch D regression tests",
                "knowledge_base_isolation": "out_of_scope: retained in Batch D regression tests",
                "inactive_documents": "out_of_scope: retained in Batch D regression tests",
                "answer_generation": "out_of_scope",
                "rerank": "out_of_scope",
            },
            started_at=started_at,
            finished_at=_utc_now(),
        )

    async def _run_query(self, query: RagF0Query) -> RagF0QueryResult:
        started = perf_counter()
        try:
            response = await self._retrieval.search(
                RagRetrievalRequest(
                    query=query.text,
                    scope=RagRetrievalScope(
                        tenant_id=RAG_F0_TENANT_ID,
                        user_id="rag-f0-evaluator",
                        agent_id="rag-f0-runner",
                        run_id=f"rag-f0-{uuid4()}",
                        tool_call_id=f"rag-f0-{query.query_id}",
                    ),
                )
            )
            if len(response.chunks) > 10:
                raise RagF0ContractError("retrieval returned more than the frozen top10")
            retrieved = []
            ranked_evidence_ids = []
            seen_evidence: set[str] = set()
            for rank, chunk in enumerate(response.chunks, 1):
                evidence = self._mapper.map_chunk(chunk)
                retrieved.append(
                    RagF0RetrievedEvidence(
                        rank=rank,
                        evidence_id=evidence.evidence_id,
                        chunk_id=chunk.id,
                        document_name=chunk.source,
                        heading_path=evidence.heading_path,
                        score=chunk.score,
                    )
                )
                if evidence.evidence_id not in seen_evidence:
                    seen_evidence.add(evidence.evidence_id)
                    ranked_evidence_ids.append(evidence.evidence_id)
            expected = self._dataset.relevant_evidence_ids(query.query_id)
            metrics = _query_metrics(
                expected_ids=expected,
                retrieved_ids=tuple(ranked_evidence_ids),
                allow_no_answer=query.allow_no_answer,
            )
            return RagF0QueryResult(
                query_id=query.query_id,
                query=query.text,
                slices=query.slices,
                allow_no_answer=query.allow_no_answer,
                latency_ms=(perf_counter() - started) * 1000,
                expected_evidence_ids=expected,
                retrieved=tuple(retrieved),
                ranked_evidence_ids=tuple(ranked_evidence_ids),
                metrics=metrics,
            )
        except RagF0ContractError:
            raise
        except Exception as exc:
            return RagF0QueryResult(
                query_id=query.query_id,
                query=query.text,
                slices=query.slices,
                allow_no_answer=query.allow_no_answer,
                latency_ms=(perf_counter() - started) * 1000,
                expected_evidence_ids=self._dataset.relevant_evidence_ids(query.query_id),
                retrieved=(),
                ranked_evidence_ids=(),
                metrics={},
                failure=exc.__class__.__name__,
            )

    def _aggregate_metrics(self, results: list[RagF0QueryResult]) -> dict[str, RagF0MetricSummary]:
        values: dict[str, list[float]] = defaultdict(list)
        for row in results:
            if row.failure is None:
                for name, value in row.metrics.items():
                    values[name].append(value)
        return {
            name: _metric_summary(
                items,
                samples=self._bootstrap_samples,
                seed=self._bootstrap_seed + index,
            )
            for index, (name, items) in enumerate(sorted(values.items()))
        }

    def _slice_metrics(
        self, results: list[RagF0QueryResult]
    ) -> dict[str, dict[str, RagF0MetricSummary]]:
        by_slice: dict[str, list[RagF0QueryResult]] = defaultdict(list)
        for row in results:
            for slice_name in row.slices:
                by_slice[slice_name].append(row)
        output = {}
        for slice_index, (slice_name, rows) in enumerate(sorted(by_slice.items())):
            values: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                if row.failure is None:
                    for name, value in row.metrics.items():
                        values[name].append(value)
            output[slice_name] = {
                name: _metric_summary(
                    items,
                    samples=self._bootstrap_samples,
                    seed=self._bootstrap_seed + 100 + slice_index * 20 + metric_index,
                )
                for metric_index, (name, items) in enumerate(sorted(values.items()))
            }
        return output


def _query_metrics(
    *, expected_ids: tuple[str, ...], retrieved_ids: tuple[str, ...], allow_no_answer: bool
) -> dict[str, float]:
    if allow_no_answer:
        return {"no_answer_false_positive_rate": 1.0 if retrieved_ids else 0.0}
    if not expected_ids:
        raise RagF0ContractError("answerable query has no expected evidence")
    metric_retrieved = list(retrieved_ids) or ["__rag_f0_no_result__"]
    expected = list(expected_ids)
    top3 = metric_retrieved[:3]
    top10 = metric_retrieved[:10]
    return {
        "recall_at_10": _score(Recall(), expected, top10),
        "hit_rate_at_3": _score(HitRate(), expected, top3),
        "mrr_at_10": _score(MRR(), expected, top10),
        "binary_ndcg_at_3": _score(NDCG(), expected, top3),
        "binary_ndcg_at_10": _score(NDCG(), expected, top10),
        "precision_at_3": _score(Precision(), expected, top3),
        "average_precision_at_10": _score(AveragePrecision(), expected, top10),
    }


def _score(metric: Any, expected_ids: list[str], retrieved_ids: list[str]) -> float:
    result = metric.compute(expected_ids=expected_ids, retrieved_ids=retrieved_ids)
    if result.score is None:
        raise RagF0ContractError(f"metric returned no score: {metric.metric_name}")
    return float(result.score)


def _metric_summary(items: list[float], *, samples: int, seed: int) -> RagF0MetricSummary:
    if not items:
        return RagF0MetricSummary(value=None, ci95_low=None, ci95_high=None, query_count=0)
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(items) for _ in range(len(items))) for _ in range(samples)
    )
    return RagF0MetricSummary(
        value=statistics.fmean(items),
        ci95_low=means[int(samples * 0.025)],
        ci95_high=means[min(samples - 1, int(samples * 0.975))],
        query_count=len(items),
    )


def _percentile(items: list[float], percentile: float) -> float | None:
    if not items:
        return None
    ordered = sorted(items)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def write_rag_f0_report(report: RagF0Report, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def infrastructure_pending_report(
    dataset: RagF0Dataset,
    failure: str,
    *,
    retrieval_mode: RagRetrievalMode = RagRetrievalMode.HYBRID,
) -> RagF0Report:
    now = _utc_now()
    return RagF0Report(
        status="infrastructure_pending",
        acceptance="pending_independent_dataset_acceptance",
        baseline=_baseline_name(retrieval_mode),
        dataset_sha256=dataset.manifest.dataset_sha256,
        dataset_version=dataset.manifest.dataset_version,
        split="dev",
        query_count=48,
        execution={"planned": 48, "executed": 0, "skipped": 48, "failed": 0},
        infrastructure_failure_count=48,
        embedding={},
        infrastructure_versions={},
        metrics={},
        slice_metrics={},
        latency_ms={"p50": None, "p95": None},
        per_query=(),
        retrieval_config={
            "hybrid_top_k": 10,
            "default_final_top_k": 3,
            "rrf_k": 60,
            "rerank_enabled": False,
            "retrieval_mode": retrieval_mode.value,
        },
        dependency_versions={
            package: version(package)
            for package in (
                "llama-index-core",
                "llama-index-vector-stores-milvus",
                "pymilvus",
                "milvus-lite",
                "openai",
            )
        },
        ablations={
            mode.value: (
                f"infrastructure_pending: {failure}"
                if mode is retrieval_mode
                else "not_executed_in_this_run"
            )
            for mode in RagRetrievalMode
        },
        out_of_scope={
            "tenant_isolation": "out_of_scope",
            "knowledge_base_isolation": "out_of_scope",
            "inactive_documents": "out_of_scope",
            "answer_generation": "out_of_scope",
            "rerank": "out_of_scope",
        },
        started_at=now,
        finished_at=now,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _baseline_name(mode: RagRetrievalMode) -> str:
    return {
        RagRetrievalMode.DENSE: "dense_only_no_rerank",
        RagRetrievalMode.BM25: "bm25_only_no_rerank",
        RagRetrievalMode.HYBRID: "dense_native_bm25_rrf_k60_no_rerank",
    }[mode]
