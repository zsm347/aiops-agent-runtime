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


REPORT_SCHEMA_VERSION = "1.1.0"
SELECTION_RULE = (
    "Use dev only. Require zero no-match false positives, zero semantic hard-negative "
    "forbidden matches, zero isolation identity/forbidden violations, and every isolation "
    "case passing. Among passing topK=3 scan points, maximize MRR, Recall@3, then HitRate@3."
)
QUALITY_GATE_RULES = (
    "full scan execution with zero infrastructure failures",
    "no-match false-positive rate equals 0",
    "semantic hard-negative forbidden evidence violations equal 0",
    "identity isolation violations equal 0",
    "isolation forbidden evidence violations equal 0",
    "all isolation contract cases pass",
)
SEMANTIC_CASE_IDS = frozenset(f"MPR{index:02d}" for index in range(1, 8))
NO_MATCH_CASE_IDS = frozenset({"MPR08", "MPR09"})
ISOLATION_CASE_IDS = frozenset({"MPR10", "MPR11", "MPR12"})


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

    @property
    def track(self) -> Literal["semantic_ranking", "no_match", "identity_isolation"]:
        if self.case_id in SEMANTIC_CASE_IDS:
            return "semantic_ranking"
        if self.case_id in NO_MATCH_CASE_IDS:
            return "no_match"
        return "identity_isolation"


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
        expected_case_ids = SEMANTIC_CASE_IDS | NO_MATCH_CASE_IDS | ISOLATION_CASE_IDS
        if (
            len(fixture_ids) != len(set(fixture_ids))
            or len(evidence_ids) != len(set(evidence_ids))
            or len(case_ids) != len(set(case_ids))
            or set(case_ids) != expected_case_ids
        ):
            raise ValueError("retrieval catalog identifiers must be unique and complete")
        evidence_catalog = set(evidence_ids)
        identity_catalog = {fixture.identity for fixture in self.fixtures}
        for case in self.cases:
            if case.identity not in identity_catalog:
                raise ValueError("retrieval case references an unknown identity")
            referenced = set(case.relevant_evidence_ids) | set(case.forbidden_evidence_ids)
            if not referenced <= evidence_catalog:
                raise ValueError("retrieval case references unknown evidence")
            if case.track == "semantic_ranking":
                if case.expected_empty or not case.relevant_evidence_ids:
                    raise ValueError("semantic ranking cases require relevant evidence")
            elif case.relevant_evidence_ids or not case.expected_empty:
                raise ValueError("no-match and isolation cases require an empty scoped result")
            if case.track == "identity_isolation" and not case.forbidden_evidence_ids:
                raise ValueError("isolation cases require a cross-identity forbidden canary")
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
class ReturnedEvidence:
    rank: int
    evidence_id: str
    similarity: float


@dataclass(frozen=True)
class CaseObservation:
    case_id: str
    category: str
    track: str
    query_sha256: str
    applied_top_k: int
    applied_min_similarity: float
    query_latency_ms: float
    returned: tuple[ReturnedEvidence, ...]
    relevant_evidence_ids: tuple[str, ...]
    forbidden_evidence_ids: tuple[str, ...]
    relevant_matches: tuple[str, ...]
    forbidden_matches: tuple[str, ...]
    unexpected_returned_ids: tuple[str, ...]
    identity_isolation_violations: int
    passed: bool


@dataclass(frozen=True)
class SemanticRankingMetrics:
    planned_cases: int
    executed_cases: int
    hit_rate_at_3: float
    recall_at_3: float
    mrr: float
    hard_negative_forbidden_violations: int


@dataclass(frozen=True)
class NoMatchMetrics:
    planned_cases: int
    executed_cases: int
    false_positive_cases: int
    false_positive_rate: float


@dataclass(frozen=True)
class IsolationMetrics:
    planned_cases: int
    executed_cases: int
    passed_cases: int
    failed_cases: int
    identity_isolation_violations: int
    forbidden_evidence_violations: int
    unexpected_result_cases: int


@dataclass(frozen=True)
class QualityGateCheck:
    name: str
    passed: bool
    actual: str
    required: str


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    checks: tuple[QualityGateCheck, ...]


@dataclass(frozen=True)
class RetrievalScanPoint:
    top_k: int
    min_similarity: float
    semantic_ranking: SemanticRankingMetrics
    no_match: NoMatchMetrics
    identity_isolation: IsolationMetrics
    quality_gate: QualityGateResult
    mean_results_per_query: float
    superset_scan_p50_latency_ms: float
    superset_scan_p95_latency_ms: float


@dataclass(frozen=True)
class CandidateEvaluation:
    top_k: int
    min_similarity: float
    accepted: bool
    semantic_ranking: SemanticRankingMetrics
    no_match: NoMatchMetrics
    identity_isolation: IsolationMetrics
    quality_gate: QualityGateResult
    observations: tuple[CaseObservation, ...]
    actual_query_p50_latency_ms: float
    actual_query_p95_latency_ms: float


@dataclass(frozen=True)
class MemoryRetrievalReport:
    report_schema_version: str
    status: str
    scan_status: str
    production_retrieval_ranking: str
    production_candidate_status: str
    dataset_version: str
    dataset_sha256: str
    embedding_provider: str
    embedding_model: str
    embedding_version: str
    embedding_dimension: int
    postgresql_version: str
    pgvector_version: str
    production_default_top_k: int
    production_default_min_similarity: float
    production_default_changed: bool
    planned_cases: int
    executed_cases: int
    skipped_cases: int
    infrastructure_failures: int
    error_codes: tuple[str, ...]
    scan_top_k_values: tuple[int, ...]
    scan_min_similarity_values: tuple[float, ...]
    superset_scan_top_k: int
    superset_scan_min_similarity: float
    selection_rule: str
    quality_gate_rules: tuple[str, ...]
    superset_scan_observations: tuple[CaseObservation, ...]
    scan: tuple[RetrievalScanPoint, ...]
    pareto_frontier: tuple[tuple[int, float], ...]
    candidate_evaluation: CandidateEvaluation | None
    selected_top_k: int | None
    selected_min_similarity: float | None
    error_analysis: tuple[dict[str, str], ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MemoryRetrievalArtifactManifest:
    manifest_schema_version: str
    report_file: str
    report_sha256: str
    dataset_sha256: str

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
    if manifest.dataset_file != dataset_path.name or manifest.dataset_sha256 != dataset_sha:
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
        production_default_top_k: int,
        production_default_min_similarity: float,
    ) -> None:
        self.repository = repository
        self.embedding_service = embedding_service
        self.dataset = dataset
        self.dataset_sha256 = dataset_sha256
        self.postgresql_version = postgresql_version
        self.pgvector_version = pgvector_version
        self.production_default_top_k = production_default_top_k
        self.production_default_min_similarity = production_default_min_similarity
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
        _validate_scan_configuration(top_k_values, min_similarity_values)
        identity = self.embedding_service.identity
        real_gate = (
            isinstance(self.repository, PostgresMemoryRepository)
            and identity.provider in REAL_MEMORY_EMBEDDING_PROVIDERS
            and identity.dimension == 1024
        )
        superset_top_k = max(top_k_values)
        superset_threshold = -1.0
        observations, scan_errors = await self._execute_cases(
            top_k=superset_top_k,
            min_similarity=superset_threshold,
        )
        scan_complete = (
            real_gate
            and len(observations) == len(self.dataset.cases)
            and not scan_errors
        )
        scan = tuple(
            self._scan_point(
                observations,
                top_k=top_k,
                min_similarity=threshold,
                scan_complete=scan_complete,
            )
            for top_k in top_k_values
            for threshold in min_similarity_values
        )
        provisional = self._select_dev_point(scan) if scan_complete else None
        candidate_evaluation: CandidateEvaluation | None = None
        candidate_errors: list[tuple[str, str]] = []
        if provisional is not None:
            candidate_observations, candidate_errors = await self._execute_cases(
                top_k=provisional.top_k,
                min_similarity=provisional.min_similarity,
            )
            candidate_evaluation = self._candidate_evaluation(
                provisional,
                candidate_observations,
                candidate_errors,
            )
        accepted = candidate_evaluation is not None and candidate_evaluation.accepted
        all_errors = scan_errors + candidate_errors
        error_analysis = self._quality_errors(
            candidate_evaluation.observations if candidate_evaluation else observations
        )
        return MemoryRetrievalReport(
            report_schema_version=REPORT_SCHEMA_VERSION,
            status="completed" if scan_complete else "failed",
            scan_status="completed" if scan_complete else "failed",
            production_retrieval_ranking="evaluated" if scan_complete else "not_evaluated",
            production_candidate_status="dev_pilot_candidate" if accepted else "no_candidate",
            dataset_version=self.dataset.version,
            dataset_sha256=self.dataset_sha256,
            embedding_provider=identity.provider,
            embedding_model=identity.model,
            embedding_version=identity.version,
            embedding_dimension=identity.dimension,
            postgresql_version=self.postgresql_version,
            pgvector_version=self.pgvector_version,
            production_default_top_k=self.production_default_top_k,
            production_default_min_similarity=self.production_default_min_similarity,
            production_default_changed=False,
            planned_cases=len(self.dataset.cases),
            executed_cases=len(observations),
            skipped_cases=len(self.dataset.cases) - len(observations),
            infrastructure_failures=len(all_errors),
            error_codes=tuple(dict.fromkeys(code for _case_id, code in all_errors)),
            scan_top_k_values=top_k_values,
            scan_min_similarity_values=min_similarity_values,
            superset_scan_top_k=superset_top_k,
            superset_scan_min_similarity=superset_threshold,
            selection_rule=SELECTION_RULE,
            quality_gate_rules=QUALITY_GATE_RULES,
            superset_scan_observations=tuple(observations),
            scan=scan,
            pareto_frontier=self._pareto_frontier(scan),
            candidate_evaluation=candidate_evaluation,
            selected_top_k=candidate_evaluation.top_k if accepted else None,
            selected_min_similarity=(
                candidate_evaluation.min_similarity if accepted else None
            ),
            error_analysis=tuple(error_analysis),
        )

    async def _execute_cases(
        self,
        *,
        top_k: int,
        min_similarity: float,
    ) -> tuple[list[CaseObservation], list[tuple[str, str]]]:
        service = MemorySearchService(
            self.repository,
            self.embedding_service,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        observations: list[CaseObservation] = []
        errors: list[tuple[str, str]] = []
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
                returned = tuple(
                    ReturnedEvidence(
                        rank=rank,
                        evidence_id=self._evidence_by_fixture[result.id],
                        similarity=result.similarity,
                    )
                    for rank, result in enumerate(results, start=1)
                )
                observations.append(
                    self._case_observation(
                        case,
                        returned,
                        top_k=top_k,
                        min_similarity=min_similarity,
                        latency_ms=(perf_counter() - started) * 1000,
                    )
                )
            except Exception as exc:
                code = (
                    exc.code
                    if isinstance(exc, MemoryPersistenceError)
                    else "memory_retrieval_infrastructure_error"
                )
                errors.append((case.case_id, code))
        return observations, errors

    def _case_observation(
        self,
        case: RetrievalCase,
        returned: tuple[ReturnedEvidence, ...],
        *,
        top_k: int,
        min_similarity: float,
        latency_ms: float,
    ) -> CaseObservation:
        returned_ids = tuple(item.evidence_id for item in returned)
        relevant_matches = tuple(
            evidence_id for evidence_id in returned_ids if evidence_id in case.relevant_evidence_ids
        )
        forbidden_matches = tuple(
            evidence_id for evidence_id in returned_ids if evidence_id in case.forbidden_evidence_ids
        )
        identity_violations = sum(
            self._fixture_by_evidence[evidence_id].identity != case.identity
            for evidence_id in returned_ids
        )
        unexpected = returned_ids if case.expected_empty else ()
        if case.track == "semantic_ranking":
            passed = bool(relevant_matches) and not forbidden_matches and identity_violations == 0
        else:
            passed = not returned_ids and not forbidden_matches and identity_violations == 0
        return CaseObservation(
            case_id=case.case_id,
            category=case.category,
            track=case.track,
            query_sha256=hashlib.sha256(case.query.encode("utf-8")).hexdigest(),
            applied_top_k=top_k,
            applied_min_similarity=min_similarity,
            query_latency_ms=latency_ms,
            returned=returned,
            relevant_evidence_ids=tuple(case.relevant_evidence_ids),
            forbidden_evidence_ids=tuple(case.forbidden_evidence_ids),
            relevant_matches=relevant_matches,
            forbidden_matches=forbidden_matches,
            unexpected_returned_ids=unexpected,
            identity_isolation_violations=identity_violations,
            passed=passed,
        )

    def _derive_observations(
        self,
        observations: list[CaseObservation],
        *,
        top_k: int,
        min_similarity: float,
    ) -> list[CaseObservation]:
        cases = {case.case_id: case for case in self.dataset.cases}
        return [
            self._case_observation(
                cases[observation.case_id],
                tuple(
                    ReturnedEvidence(rank=index, evidence_id=item.evidence_id, similarity=item.similarity)
                    for index, item in enumerate(
                        (
                            item
                            for item in observation.returned
                            if item.similarity >= min_similarity
                        ),
                        start=1,
                    )
                    if index <= top_k
                ),
                top_k=top_k,
                min_similarity=min_similarity,
                latency_ms=observation.query_latency_ms,
            )
            for observation in observations
        ]

    def _scan_point(
        self,
        observations: list[CaseObservation],
        *,
        top_k: int,
        min_similarity: float,
        scan_complete: bool = True,
    ) -> RetrievalScanPoint:
        derived = self._derive_observations(
            observations,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        semantic, no_match, isolation = _track_metrics(derived)
        gate = _quality_gate(
            semantic,
            no_match,
            isolation,
            full_execution=scan_complete and len(derived) == len(self.dataset.cases),
        )
        latencies = [observation.query_latency_ms for observation in observations]
        returned_count = sum(len(observation.returned) for observation in derived)
        return RetrievalScanPoint(
            top_k=top_k,
            min_similarity=min_similarity,
            semantic_ranking=semantic,
            no_match=no_match,
            identity_isolation=isolation,
            quality_gate=gate,
            mean_results_per_query=returned_count / (len(derived) or 1),
            superset_scan_p50_latency_ms=_percentile(latencies, 0.50),
            superset_scan_p95_latency_ms=_percentile(latencies, 0.95),
        )

    @staticmethod
    def _select_dev_point(scan: tuple[RetrievalScanPoint, ...]) -> RetrievalScanPoint | None:
        candidates = [
            point for point in scan if point.top_k == 3 and point.quality_gate.passed
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda point: (
                point.semantic_ranking.mrr,
                point.semantic_ranking.recall_at_3,
                point.semantic_ranking.hit_rate_at_3,
                -point.min_similarity,
            ),
        )

    def _candidate_evaluation(
        self,
        provisional: RetrievalScanPoint,
        observations: list[CaseObservation],
        errors: list[tuple[str, str]],
    ) -> CandidateEvaluation:
        semantic, no_match, isolation = _track_metrics(observations)
        gate = _quality_gate(
            semantic,
            no_match,
            isolation,
            full_execution=not errors and len(observations) == len(self.dataset.cases),
        )
        latencies = [observation.query_latency_ms for observation in observations]
        return CandidateEvaluation(
            top_k=provisional.top_k,
            min_similarity=provisional.min_similarity,
            accepted=gate.passed,
            semantic_ranking=semantic,
            no_match=no_match,
            identity_isolation=isolation,
            quality_gate=gate,
            observations=tuple(observations),
            actual_query_p50_latency_ms=_percentile(latencies, 0.50),
            actual_query_p95_latency_ms=_percentile(latencies, 0.95),
        )

    @staticmethod
    def _pareto_frontier(
        scan: tuple[RetrievalScanPoint, ...],
    ) -> tuple[tuple[int, float], ...]:
        points = [point for point in scan if point.top_k == 3]
        frontier: list[tuple[int, float]] = []
        for candidate in points:
            dominated = any(
                _dominates(other, candidate) for other in points if other is not candidate
            )
            if not dominated:
                frontier.append((candidate.top_k, candidate.min_similarity))
        return tuple(frontier)

    def _quality_errors(
        self,
        observations: tuple[CaseObservation, ...] | list[CaseObservation],
    ) -> list[dict[str, str]]:
        errors: list[dict[str, str]] = []
        for observation in observations:
            codes: list[str] = []
            if observation.track == "semantic_ranking" and not observation.relevant_matches:
                codes.append("relevant_evidence_missed")
            elif (
                observation.track == "semantic_ranking"
                and set(observation.relevant_matches) != set(observation.relevant_evidence_ids)
            ):
                codes.append("relevant_evidence_partial")
            if observation.forbidden_matches:
                codes.append("forbidden_evidence_returned")
            if observation.unexpected_returned_ids:
                codes.append("unexpected_nonempty_result")
            if observation.identity_isolation_violations:
                codes.append("identity_isolation_violation")
            for code in codes:
                errors.append(
                    {
                        "case_id": observation.case_id,
                        "category": observation.category,
                        "track": observation.track,
                        "error_code": code,
                    }
                )
        return errors


def _track_metrics(
    observations: list[CaseObservation],
) -> tuple[SemanticRankingMetrics, NoMatchMetrics, IsolationMetrics]:
    semantic = [item for item in observations if item.track == "semantic_ranking"]
    no_match_items = [item for item in observations if item.track == "no_match"]
    isolation_items = [item for item in observations if item.track == "identity_isolation"]
    hits = recalls = reciprocal_ranks = 0.0
    hard_negative_forbidden = 0
    for observation in semantic:
        returned_ids = tuple(item.evidence_id for item in observation.returned[:3])
        relevant = set(observation.relevant_evidence_ids)
        matches = relevant.intersection(returned_ids)
        hits += float(bool(matches))
        recalls += len(matches) / len(relevant)
        reciprocal_ranks += next(
            (
                1.0 / rank
                for rank, evidence_id in enumerate(returned_ids, start=1)
                if evidence_id in relevant
            ),
            0.0,
        )
        if observation.category == "hard_negative":
            hard_negative_forbidden += len(observation.forbidden_matches)
    semantic_denominator = len(SEMANTIC_CASE_IDS)
    no_match_false_positives = sum(bool(item.returned) for item in no_match_items)
    isolation_identity_violations = sum(
        item.identity_isolation_violations for item in isolation_items
    )
    isolation_forbidden = sum(len(item.forbidden_matches) for item in isolation_items)
    isolation_passed = sum(item.passed for item in isolation_items)
    return (
        SemanticRankingMetrics(
            planned_cases=len(SEMANTIC_CASE_IDS),
            executed_cases=len(semantic),
            hit_rate_at_3=hits / semantic_denominator,
            recall_at_3=recalls / semantic_denominator,
            mrr=reciprocal_ranks / semantic_denominator,
            hard_negative_forbidden_violations=hard_negative_forbidden,
        ),
        NoMatchMetrics(
            planned_cases=len(NO_MATCH_CASE_IDS),
            executed_cases=len(no_match_items),
            false_positive_cases=no_match_false_positives,
            false_positive_rate=no_match_false_positives / len(NO_MATCH_CASE_IDS),
        ),
        IsolationMetrics(
            planned_cases=len(ISOLATION_CASE_IDS),
            executed_cases=len(isolation_items),
            passed_cases=isolation_passed,
            failed_cases=len(ISOLATION_CASE_IDS) - isolation_passed,
            identity_isolation_violations=isolation_identity_violations,
            forbidden_evidence_violations=isolation_forbidden,
            unexpected_result_cases=sum(bool(item.returned) for item in isolation_items),
        ),
    )


def _quality_gate(
    semantic: SemanticRankingMetrics,
    no_match: NoMatchMetrics,
    isolation: IsolationMetrics,
    *,
    full_execution: bool,
) -> QualityGateResult:
    checks = (
        QualityGateCheck("full_execution", full_execution, str(full_execution), "True"),
        QualityGateCheck(
            "no_match_false_positive_rate",
            no_match.false_positive_rate == 0.0,
            str(no_match.false_positive_rate),
            "0.0",
        ),
        QualityGateCheck(
            "hard_negative_forbidden_violations",
            semantic.hard_negative_forbidden_violations == 0,
            str(semantic.hard_negative_forbidden_violations),
            "0",
        ),
        QualityGateCheck(
            "identity_isolation_violations",
            isolation.identity_isolation_violations == 0,
            str(isolation.identity_isolation_violations),
            "0",
        ),
        QualityGateCheck(
            "isolation_forbidden_evidence_violations",
            isolation.forbidden_evidence_violations == 0,
            str(isolation.forbidden_evidence_violations),
            "0",
        ),
        QualityGateCheck(
            "isolation_cases_pass",
            isolation.passed_cases == isolation.planned_cases,
            f"{isolation.passed_cases}/{isolation.planned_cases}",
            f"{isolation.planned_cases}/{isolation.planned_cases}",
        ),
    )
    return QualityGateResult(passed=all(check.passed for check in checks), checks=checks)


def _dominates(left: RetrievalScanPoint, right: RetrievalScanPoint) -> bool:
    left_values = (
        left.semantic_ranking.hit_rate_at_3,
        left.semantic_ranking.recall_at_3,
        left.semantic_ranking.mrr,
        -left.no_match.false_positive_rate,
        -left.semantic_ranking.hard_negative_forbidden_violations,
        -left.identity_isolation.failed_cases,
    )
    right_values = (
        right.semantic_ranking.hit_rate_at_3,
        right.semantic_ranking.recall_at_3,
        right.semantic_ranking.mrr,
        -right.no_match.false_positive_rate,
        -right.semantic_ranking.hard_negative_forbidden_violations,
        -right.identity_isolation.failed_cases,
    )
    return all(left >= right for left, right in zip(left_values, right_values, strict=True)) and any(
        left > right for left, right in zip(left_values, right_values, strict=True)
    )


def _validate_scan_configuration(
    top_k_values: tuple[int, ...],
    min_similarity_values: tuple[float, ...],
) -> None:
    if (
        not top_k_values
        or 3 not in top_k_values
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in top_k_values)
        or len(set(top_k_values)) != len(top_k_values)
        or not min_similarity_values
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < -1.0
            or value > 1.0
            for value in min_similarity_values
        )
        or len(set(min_similarity_values)) != len(min_similarity_values)
    ):
        raise ValueError("memory retrieval scan configuration is invalid")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def serialize_memory_retrieval_report(report: MemoryRetrievalReport) -> bytes:
    return (
        json.dumps(
            report.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_memory_retrieval_report(
    report: MemoryRetrievalReport,
    path: Path,
) -> tuple[MemoryRetrievalArtifactManifest, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = serialize_memory_retrieval_report(report)
    path.write_bytes(report_bytes)
    manifest = MemoryRetrievalArtifactManifest(
        manifest_schema_version="1.0.0",
        report_file=path.name,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        dataset_sha256=report.dataset_sha256,
    )
    manifest_path = path.with_name(f"{path.stem}.manifest.json")
    manifest_path.write_text(
        json.dumps(
            manifest.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, manifest_path
