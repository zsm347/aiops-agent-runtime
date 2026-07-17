from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from superbiz_agent.memory.dedup import canonical_content_hash
from superbiz_agent.memory.embedding import (
    REAL_MEMORY_EMBEDDING_PROVIDERS,
    MemoryEmbeddingService,
)
from superbiz_agent.memory.errors import MemoryPersistenceError
from superbiz_agent.memory.ports import MemoryRepository
from superbiz_agent.memory.schemas import LongTermMemory
from superbiz_agent.memory.search import MemorySearchService
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository


class RetrievalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RetrievalFixture(RetrievalModel):
    fixture_id: str = Field(pattern=r"^mpr1-fixture-[a-z0-9-]+$")
    evidence_id: str = Field(pattern=r"^mpr1-evidence-[a-z0-9-]+$")
    identity: str
    type: Literal["experience", "knowledge"]
    topic: str
    content: str
    tags: list[str] = Field(default_factory=list)
    scope_service: str | None = None
    scope_env: str | None = None


class RetrievalCase(RetrievalModel):
    case_id: str = Field(pattern=r"^MPR[0-9]{2}$")
    category: Literal[
        "positive",
        "paraphrase",
        "hard_negative",
        "no_match",
        "scope_filter",
        "identity_isolation",
    ]
    identity: str
    query: str
    relevant_evidence_ids: list[str]
    forbidden_evidence_ids: list[str] = Field(default_factory=list)
    expected_empty: bool = False
    optional_type: Literal["experience", "knowledge"] | None = None
    scope_service: str | None = None
    scope_env: str | None = None
    tags: list[str] = Field(default_factory=list)


class MemoryRetrievalDataset(RetrievalModel):
    version: str
    split: Literal["dev"]
    fixtures: list[RetrievalFixture]
    cases: list[RetrievalCase]

    @model_validator(mode="after")
    def validate_catalog(self) -> "MemoryRetrievalDataset":
        fixture_ids = [fixture.fixture_id for fixture in self.fixtures]
        evidence_ids = [fixture.evidence_id for fixture in self.fixtures]
        case_ids = [case.case_id for case in self.cases]
        if (
            len(fixture_ids) != len(set(fixture_ids))
            or len(evidence_ids) != len(set(evidence_ids))
            or len(case_ids) != len(set(case_ids))
        ):
            raise ValueError("retrieval catalog identifiers must be unique")
        evidence_catalog = set(evidence_ids)
        identity_catalog = {fixture.identity for fixture in self.fixtures}
        for case in self.cases:
            if case.identity not in identity_catalog:
                raise ValueError("retrieval case references an unknown identity")
            referenced = set(case.relevant_evidence_ids) | set(case.forbidden_evidence_ids)
            if not referenced <= evidence_catalog:
                raise ValueError("retrieval case references unknown evidence")
            if case.expected_empty != (case.category == "no_match"):
                raise ValueError("only no-match cases may require an empty result")
            if not case.expected_empty and not case.relevant_evidence_ids:
                raise ValueError("ranking cases require relevant evidence")
        return self


class MemoryRetrievalManifest(RetrievalModel):
    version: str
    split: Literal["dev"]
    dataset_file: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_evidence: dict[str, str]


@dataclass(frozen=True)
class RetrievalIdentity:
    tenant_id: str
    user_id: str
    agent_id: str


@dataclass(frozen=True)
class RetrievalObservation:
    case_id: str
    category: str
    evidence_ids: tuple[str, ...]
    similarities: tuple[float, ...]
    latency_ms: float


@dataclass(frozen=True)
class RetrievalScanPoint:
    top_k: int
    min_similarity: float
    hit_rate_at_3: float
    recall_at_3: float
    mrr: float
    no_match_false_positive_rate: float
    identity_isolation_violations: int
    forbidden_result_violations: int
    mean_results_per_query: float
    p50_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True)
class MemoryRetrievalReport:
    status: str
    production_retrieval_ranking: str
    dataset_version: str
    dataset_sha256: str
    embedding_provider: str
    embedding_model: str
    embedding_version: str
    embedding_dimension: int
    postgresql_version: str
    pgvector_version: str
    planned_cases: int
    executed_cases: int
    skipped_cases: int
    infrastructure_failures: int
    error_codes: tuple[str, ...]
    scan: tuple[RetrievalScanPoint, ...]
    selected_top_k: int | None
    selected_min_similarity: float | None
    error_analysis: tuple[dict[str, str], ...]

    def to_dict(self) -> dict:
        return asdict(self)


IDENTITIES = {
    "primary": RetrievalIdentity("mpr1-tenant", "mpr1-user", "mpr1-agent"),
    "other_tenant": RetrievalIdentity("mpr1-other-tenant", "mpr1-user", "mpr1-agent"),
    "other_user": RetrievalIdentity("mpr1-tenant", "mpr1-other-user", "mpr1-agent"),
    "other_agent": RetrievalIdentity("mpr1-tenant", "mpr1-user", "mpr1-other-agent"),
}


def load_memory_retrieval_dataset(
    dataset_path: Path,
    manifest_path: Path,
) -> tuple[MemoryRetrievalDataset, str]:
    dataset_bytes = dataset_path.read_bytes()
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    manifest = MemoryRetrievalManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.dataset_file != dataset_path.name
        or manifest.dataset_sha256 != dataset_sha
        or manifest.version != "1.0.0"
    ):
        raise ValueError("memory retrieval manifest does not match the dataset")
    dataset = MemoryRetrievalDataset.model_validate_json(dataset_bytes)
    fixture_map = {fixture.fixture_id: fixture.evidence_id for fixture in dataset.fixtures}
    if fixture_map != manifest.fixture_evidence or dataset.version != manifest.version:
        raise ValueError("memory retrieval fixture manifest does not match the dataset")
    return dataset, dataset_sha


class MemoryRetrievalRunner:
    def __init__(
        self,
        repository: MemoryRepository,
        embedding_service: MemoryEmbeddingService,
        dataset: MemoryRetrievalDataset,
        dataset_sha256: str,
        *,
        postgresql_version: str,
        pgvector_version: str,
    ) -> None:
        self.repository = repository
        self.embedding_service = embedding_service
        self.dataset = dataset
        self.dataset_sha256 = dataset_sha256
        self.postgresql_version = postgresql_version
        self.pgvector_version = pgvector_version
        self._evidence_by_fixture = {
            fixture.fixture_id: fixture.evidence_id for fixture in dataset.fixtures
        }
        self._fixture_by_evidence = {
            fixture.evidence_id: fixture for fixture in dataset.fixtures
        }

    async def seed_fixtures(self) -> None:
        contents = tuple(fixture.content for fixture in self.dataset.fixtures)
        batch = await self.embedding_service.embed_documents(contents)
        if batch.identity != self.embedding_service.identity or len(batch.vectors) != len(contents):
            raise ValueError("embedding fixture batch violated the memory contract")
        for fixture, vector in zip(self.dataset.fixtures, batch.vectors, strict=True):
            identity = IDENTITIES[fixture.identity]
            result = await self.repository.write_archival_exact(
                LongTermMemory(
                    id=fixture.fixture_id,
                    tenant_id=identity.tenant_id,
                    user_id=identity.user_id,
                    agent_id=identity.agent_id,
                    session_id="mpr1-retrieval-baseline",
                    type=fixture.type,
                    topic=fixture.topic,
                    content=fixture.content,
                    embedding=list(vector),
                    content_hash=canonical_content_hash(fixture.content),
                    source="manual",
                    embedding_provider=batch.identity.provider,
                    embedding_model=batch.identity.model,
                    embedding_version=batch.identity.version,
                    embedding_dimension=batch.identity.dimension,
                    tags=list(fixture.tags),
                    scope_service=fixture.scope_service,
                    scope_env=fixture.scope_env,
                )
            )
            if result.status != "written" or result.memory.id != fixture.fixture_id:
                raise ValueError("retrieval fixture write was not isolated and fresh")

    async def run(
        self,
        *,
        top_k_values: tuple[int, ...] = (1, 3, 5),
        min_similarity_values: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7),
    ) -> MemoryRetrievalReport:
        identity = self.embedding_service.identity
        real_gate = (
            isinstance(self.repository, PostgresMemoryRepository)
            and identity.provider in REAL_MEMORY_EMBEDDING_PROVIDERS
            and identity.dimension == 1024
        )
        observations: list[RetrievalObservation] = []
        infrastructure_failures = 0
        error_codes: list[str] = []
        error_analysis: list[dict[str, str]] = []
        service = MemorySearchService(
            self.repository,
            self.embedding_service,
            top_k=max(top_k_values),
            min_similarity=-1.0,
        )

        for case in self.dataset.cases:
            case_identity = IDENTITIES[case.identity]
            started = perf_counter()
            try:
                results = await service.search_memory(
                    case_identity.tenant_id,
                    case_identity.user_id,
                    case_identity.agent_id,
                    case.query,
                    case.optional_type,
                    scope_service=case.scope_service,
                    scope_env=case.scope_env,
                    tags=case.tags,
                )
                observations.append(
                    RetrievalObservation(
                        case_id=case.case_id,
                        category=case.category,
                        evidence_ids=tuple(self._evidence_by_fixture[result.id] for result in results),
                        similarities=tuple(result.similarity for result in results),
                        latency_ms=(perf_counter() - started) * 1000,
                    )
                )
            except Exception as exc:
                infrastructure_failures += 1
                code = (
                    exc.code
                    if isinstance(exc, MemoryPersistenceError)
                    else "memory_retrieval_infrastructure_error"
                )
                error_codes.append(code)
                error_analysis.append({"case_id": case.case_id, "error_code": code})

        scan = tuple(
            self._scan_point(observations, top_k=top_k, min_similarity=threshold)
            for top_k in top_k_values
            for threshold in min_similarity_values
        )
        selected = self._select_dev_point(scan)
        if selected is not None:
            error_analysis.extend(self._quality_errors(observations, selected))
        executed = len(observations)
        evaluated = (
            real_gate
            and executed == len(self.dataset.cases)
            and infrastructure_failures == 0
        )
        return MemoryRetrievalReport(
            status="completed" if evaluated else "failed",
            production_retrieval_ranking="evaluated" if evaluated else "not_evaluated",
            dataset_version=self.dataset.version,
            dataset_sha256=self.dataset_sha256,
            embedding_provider=identity.provider,
            embedding_model=identity.model,
            embedding_version=identity.version,
            embedding_dimension=identity.dimension,
            postgresql_version=self.postgresql_version,
            pgvector_version=self.pgvector_version,
            planned_cases=len(self.dataset.cases),
            executed_cases=executed,
            skipped_cases=len(self.dataset.cases) - executed,
            infrastructure_failures=infrastructure_failures,
            error_codes=tuple(dict.fromkeys(error_codes)),
            scan=scan,
            selected_top_k=selected.top_k if selected else None,
            selected_min_similarity=selected.min_similarity if selected else None,
            error_analysis=tuple(error_analysis),
        )

    def _scan_point(
        self,
        observations: list[RetrievalObservation],
        *,
        top_k: int,
        min_similarity: float,
    ) -> RetrievalScanPoint:
        by_case = {observation.case_id: observation for observation in observations}
        ranking_cases = [case for case in self.dataset.cases if case.relevant_evidence_ids]
        hits = recalls = reciprocal_ranks = 0.0
        no_match_total = no_match_false_positives = 0
        identity_violations = forbidden_violations = returned_total = 0
        latencies = [observation.latency_ms for observation in observations]

        for case in self.dataset.cases:
            observation = by_case.get(case.case_id)
            if observation is None:
                continue
            evidence = tuple(
                evidence_id
                for evidence_id, similarity in zip(
                    observation.evidence_ids,
                    observation.similarities,
                    strict=True,
                )
                if similarity >= min_similarity
            )[:top_k]
            returned_total += len(evidence)
            relevant = set(case.relevant_evidence_ids)
            if relevant:
                top_three = evidence[:3]
                matched = relevant.intersection(top_three)
                hits += float(bool(matched))
                recalls += len(matched) / len(relevant)
                reciprocal_ranks += next(
                    (1.0 / rank for rank, value in enumerate(evidence, start=1) if value in relevant),
                    0.0,
                )
            if case.expected_empty:
                no_match_total += 1
                no_match_false_positives += int(bool(evidence))
            forbidden_violations += len(set(evidence).intersection(case.forbidden_evidence_ids))
            for evidence_id in evidence:
                fixture = self._fixture_by_evidence[evidence_id]
                identity_violations += int(fixture.identity != case.identity)

        denominator = len(ranking_cases) or 1
        return RetrievalScanPoint(
            top_k=top_k,
            min_similarity=min_similarity,
            hit_rate_at_3=hits / denominator,
            recall_at_3=recalls / denominator,
            mrr=reciprocal_ranks / denominator,
            no_match_false_positive_rate=(
                no_match_false_positives / no_match_total if no_match_total else 0.0
            ),
            identity_isolation_violations=identity_violations,
            forbidden_result_violations=forbidden_violations,
            mean_results_per_query=returned_total / (len(observations) or 1),
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
        )

    @staticmethod
    def _select_dev_point(scan: tuple[RetrievalScanPoint, ...]) -> RetrievalScanPoint | None:
        candidates = [point for point in scan if point.top_k == 3]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda point: (
                -point.no_match_false_positive_rate,
                point.mrr,
                point.recall_at_3,
                -point.forbidden_result_violations,
                point.min_similarity,
            ),
        )

    def _quality_errors(
        self,
        observations: list[RetrievalObservation],
        point: RetrievalScanPoint,
    ) -> list[dict[str, str]]:
        cases = {case.case_id: case for case in self.dataset.cases}
        errors: list[dict[str, str]] = []
        for observation in observations:
            case = cases[observation.case_id]
            evidence = tuple(
                evidence_id
                for evidence_id, similarity in zip(
                    observation.evidence_ids,
                    observation.similarities,
                    strict=True,
                )
                if similarity >= point.min_similarity
            )[: point.top_k]
            relevant = set(case.relevant_evidence_ids)
            matched = relevant.intersection(evidence[:3])
            codes: list[str] = []
            if case.expected_empty and evidence:
                codes.append("no_match_false_positive")
            if relevant and not matched:
                codes.append("relevant_evidence_missed")
            elif relevant and matched != relevant:
                codes.append("relevant_evidence_partial")
            if set(evidence).intersection(case.forbidden_evidence_ids):
                codes.append("forbidden_evidence_returned")
            for code in codes:
                errors.append(
                    {
                        "case_id": case.case_id,
                        "category": case.category,
                        "error_code": code,
                    }
                )
        return errors


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def write_memory_retrieval_report(report: MemoryRetrievalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
