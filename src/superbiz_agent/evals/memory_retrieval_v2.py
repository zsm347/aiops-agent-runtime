"""M-P1-R2 retrieval dataset, redacted observations and offline calibration.

This module is intentionally independent from the frozen v1 evaluator.  It
does not select or persist a production threshold; it only describes dev-set
feasibility and Pareto trade-offs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from superbiz_agent.memory.dedup import canonical_content_hash
from superbiz_agent.memory.embedding import MemoryEmbeddingService
from superbiz_agent.memory.errors import MemoryPersistenceError
from superbiz_agent.memory.ports import MemoryRepository
from superbiz_agent.memory.schemas import LongTermMemory
from superbiz_agent.memory.search import MemorySearchService


V2_REPORT_SCHEMA_VERSION = "2.0.0"
V2_MANIFEST_SCHEMA_VERSION = "1.0.0"
V2_IDENTITY_MAP = {
    "primary": ("mpr2-tenant", "mpr2-user", "mpr2-agent"),
    "other_tenant": ("mpr2-other-tenant", "mpr2-user", "mpr2-agent"),
    "other_user": ("mpr2-tenant", "mpr2-other-user", "mpr2-agent"),
    "other_agent": ("mpr2-tenant", "mpr2-user", "mpr2-other-agent"),
}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


class V2Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class V2Fixture(V2Model):
    fixture_id: str
    evidence_id: str
    identity: Literal["primary", "other_tenant", "other_user", "other_agent"]
    type: Literal["experience", "knowledge"]
    topic: str
    content: str
    tags: list[str] = Field(default_factory=list)
    scope_service: str | None = None
    scope_env: str | None = None


class V2Query(V2Model):
    case_id: str
    split: Literal["dev", "holdout"]
    track: Literal["semantic_ranking", "no_match", "identity_isolation"]
    category: str
    slices: list[str]
    identity: Literal["primary", "other_tenant", "other_user", "other_agent"]
    identity_axis: Literal["tenant", "user", "agent"] | None = None
    query: str
    relevant_evidence_ids: list[str] = Field(default_factory=list)
    qrel_rationales: dict[str, str] = Field(default_factory=dict)
    forbidden_evidence_ids: list[str] = Field(default_factory=list)
    hard_negative_rationales: dict[str, str] = Field(default_factory=dict)
    expected_empty: bool = False
    optional_type: Literal["experience", "knowledge"] | None = None
    scope_service: str | None = None
    scope_env: str | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_semantics(self) -> "V2Query":
        if self.track == "semantic_ranking" and (self.expected_empty or not self.relevant_evidence_ids):
            raise ValueError("semantic query requires relevant evidence")
        if self.track != "semantic_ranking" and (self.relevant_evidence_ids or not self.expected_empty):
            raise ValueError("non-semantic query must be expected empty without qrels")
        if self.track == "identity_isolation" and (self.identity_axis is None or not self.forbidden_evidence_ids):
            raise ValueError("isolation query requires an identity axis and forbidden canary")
        if set(self.relevant_evidence_ids) - set(self.qrel_rationales):
            raise ValueError("every qrel requires a rationale")
        if set(self.forbidden_evidence_ids) - set(self.hard_negative_rationales):
            raise ValueError("every forbidden evidence requires a rationale")
        return self


class V2Dataset(V2Model):
    version: str
    name: str
    provenance: str
    fixtures: list[V2Fixture]
    queries: list[V2Query]

    @model_validator(mode="after")
    def validate_catalog(self) -> "V2Dataset":
        fixture_ids = [f.fixture_id for f in self.fixtures]
        evidence_ids = [f.evidence_id for f in self.fixtures]
        query_ids = [q.case_id for q in self.queries]
        if len(fixture_ids) != len(set(fixture_ids)) or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("fixture identifiers must be unique")
        if len(query_ids) != len(set(query_ids)) or len(self.fixtures) != 60 or len(self.queries) != 60:
            raise ValueError("v2 catalog must contain 60 fixtures and 60 queries")
        evidence_set = set(evidence_ids)
        fixture_by_evidence = {fixture.evidence_id: fixture for fixture in self.fixtures}
        for query in self.queries:
            refs = set(query.relevant_evidence_ids) | set(query.forbidden_evidence_ids)
            if not refs <= evidence_set:
                raise ValueError("query references unknown evidence")
            if query.track == "identity_isolation":
                expected_identity = {
                    "tenant": "other_tenant",
                    "user": "other_user",
                    "agent": "other_agent",
                }[query.identity_axis]
                if any(
                    fixture_by_evidence[evidence_id].identity != expected_identity
                    for evidence_id in query.forbidden_evidence_ids
                ):
                    raise ValueError("isolation canary does not match the declared identity axis")
        dev = [q for q in self.queries if q.split == "dev"]
        holdout = [q for q in self.queries if q.split == "holdout"]
        if len(dev) != 48 or len(holdout) != 12:
            raise ValueError("v2 split must contain 48 dev and 12 holdout queries")
        if Counter(q.track for q in dev) != Counter({"semantic_ranking": 30, "no_match": 10, "identity_isolation": 8}):
            raise ValueError("v2 dev tracks are incomplete")
        if Counter(f.identity for f in self.fixtures) != Counter({"primary": 51, "other_tenant": 3, "other_user": 3, "other_agent": 3}):
            raise ValueError("v2 identity fixture counts are incomplete")
        return self


@dataclass(frozen=True)
class V2ReturnedEvidence:
    rank: int
    evidence_id: str
    similarity: float


@dataclass(frozen=True)
class V2Observation:
    case_id: str
    split: str
    track: str
    category: str
    query_sha256: str
    returned: tuple[V2ReturnedEvidence, ...]
    relevant_evidence_ids: tuple[str, ...]
    forbidden_evidence_ids: tuple[str, ...]
    relevant_matches: tuple[str, ...]
    forbidden_matches: tuple[str, ...]
    identity_leakage: int
    unexpected_result: bool
    query_latency_ms: float
    top1_similarity: float | None
    top2_similarity: float | None
    top1_top2_margin: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackMetrics:
    planned: int
    executed: int
    hit_rate_at_1: float
    hit_rate_at_3: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    hard_negative_forbidden_violations: int
    false_positive_count: int
    false_positive_rate: float
    identity_leakage: int
    forbidden_evidence_violations: int
    unexpected_result_cases: int
    empty_result_pass_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateResult:
    accepted_cases: tuple[str, ...]
    rejected_cases: tuple[str, ...]
    answerable_acceptance: float
    answerable_false_rejection: float
    no_answer_fpr: float
    isolation_empty_rate: float
    semantic_hit_rate_at_3: float
    semantic_recall_at_3: float
    semantic_mrr: float
    identity_leakage: int
    false_accepted_case_ids: tuple[str, ...]
    false_rejected_case_ids: tuple[str, ...]
    feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationPoint:
    strategy: str
    top1_threshold: float
    margin_threshold: float | None
    gate: GateResult

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "top1_threshold": self.top1_threshold, "margin_threshold": self.margin_threshold, **self.gate.to_dict()}


def load_memory_retrieval_v2_dataset(dataset_path: Path, manifest_path: Path) -> tuple[V2Dataset, str]:
    raw = dataset_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_file") != dataset_path.name or manifest.get("dataset_sha256") != sha:
        raise ValueError("v2 dataset manifest does not match dataset bytes")
    dataset = V2Dataset.model_validate_json(raw)
    if manifest.get("query_ids") != [q.case_id for q in dataset.queries]:
        raise ValueError("v2 query manifest does not match dataset")
    return dataset, sha


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def analyze_memory_retrieval_v2_quality(dataset: V2Dataset) -> dict[str, Any]:
    fixture_ids = [f.fixture_id for f in dataset.fixtures]
    evidence_ids = [f.evidence_id for f in dataset.fixtures]
    queries = dataset.queries
    dev = [q for q in queries if q.split == "dev"]
    holdout = [q for q in queries if q.split == "holdout"]
    max_cross_split = 0.0
    nearest_pairs: list[dict[str, Any]] = []
    for left in dev:
        for right in holdout:
            left_tokens, right_tokens = _tokens(left.query), _tokens(right.query)
            score = len(left_tokens & right_tokens) / (len(left_tokens | right_tokens) or 1)
            if score > max_cross_split:
                max_cross_split = score
            if score >= 0.8:
                nearest_pairs.append({"dev_case_id": left.case_id, "holdout_case_id": right.case_id, "jaccard": round(score, 6)})
    overlap = []
    evidence_by_id = {f.evidence_id: f for f in dataset.fixtures}
    for query in dev:
        q_tokens = _tokens(query.query)
        gold_tokens = set().union(*(_tokens(evidence_by_id[e].content) for e in query.relevant_evidence_ids)) if query.relevant_evidence_ids else set()
        overlap.append(len(q_tokens & gold_tokens) / (len(q_tokens) or 1))
    return {
        "status": "pending independent dataset acceptance",
        "dataset_version": dataset.version,
        "fixture_count": len(dataset.fixtures),
        "query_count": len(queries),
        "split_counts": {"dev": len(dev), "holdout": len(holdout)},
        "track_counts": dict(Counter(q.track for q in dev)),
        "slice_counts": dict(Counter(slice_name for query in dev for slice_name in query.slices)),
        "identity_fixture_counts": dict(Counter(f.identity for f in dataset.fixtures)),
        "fixture_ids_unique": len(fixture_ids) == len(set(fixture_ids)),
        "evidence_ids_unique": len(evidence_ids) == len(set(evidence_ids)),
        "query_ids_unique": len({q.case_id for q in queries}) == len(queries),
        "query_texts_unique": len({" ".join(sorted(_tokens(q.query))) for q in queries}) == len(queries),
        "isolation_axis_valid": all(
            all(
                evidence_by_id[evidence_id].identity
                == {"tenant": "other_tenant", "user": "other_user", "agent": "other_agent"}[
                    query.identity_axis
                ]
                for evidence_id in query.forbidden_evidence_ids
            )
            for query in queries
            if query.track == "identity_isolation"
        ),
        "all_qrels_have_rationale": all(set(q.relevant_evidence_ids) <= set(q.qrel_rationales) for q in queries),
        "all_hard_negatives_have_rationale": all(set(q.forbidden_evidence_ids) <= set(q.hard_negative_rationales) for q in queries),
        "cross_split_near_duplicates": nearest_pairs,
        "max_cross_split_jaccard": round(max_cross_split, 6),
        "query_gold_overlap_min": round(min(overlap), 6) if overlap else 0.0,
        "query_gold_overlap_max": round(max(overlap), 6) if overlap else 0.0,
        "query_gold_overlap_mean": round(sum(overlap) / len(overlap), 6) if overlap else 0.0,
        "canonical_content_hash_unique": len({canonical_content_hash(f.content) for f in dataset.fixtures}) == len(dataset.fixtures),
        "quality_gate": (
            not nearest_pairs
            and len(fixture_ids) == len(set(fixture_ids))
            and len(evidence_ids) == len(set(evidence_ids))
            and len({q.case_id for q in queries}) == len(queries)
            and len({" ".join(sorted(_tokens(q.query))) for q in queries}) == len(queries)
            and all(set(q.relevant_evidence_ids) <= set(q.qrel_rationales) for q in queries)
            and all(set(q.forbidden_evidence_ids) <= set(q.hard_negative_rationales) for q in queries)
            and len({canonical_content_hash(f.content) for f in dataset.fixtures})
            == len(dataset.fixtures)
        ),
    }


class MemoryRetrievalV2Runner:
    """Seed fixtures and execute only dev queries against production retrieval."""

    def __init__(self, repository: MemoryRepository, embedding_service: MemoryEmbeddingService, dataset: V2Dataset, dataset_sha256: str, *, postgresql_version: str, pgvector_version: str) -> None:
        self.repository = repository
        self.embedding_service = embedding_service
        self.dataset = dataset
        self.dataset_sha256 = dataset_sha256
        self.postgresql_version = postgresql_version
        self.pgvector_version = pgvector_version
        self._fixtures = {f.evidence_id: f for f in dataset.fixtures}
        self._evidence_by_fixture = {f.fixture_id: f.evidence_id for f in dataset.fixtures}

    async def seed_fixtures(self) -> tuple[str, ...]:
        fixtures = self.dataset.fixtures
        batch = await self.embedding_service.embed_documents([f.content for f in fixtures])
        if batch.identity != self.embedding_service.identity or len(batch.vectors) != len(fixtures):
            raise ValueError("v2 fixture embedding contract violated")
        ids: list[str] = []
        for fixture, vector in zip(fixtures, batch.vectors, strict=True):
            tenant, user, agent = V2_IDENTITY_MAP[fixture.identity]
            result = await self.repository.write_archival_exact(LongTermMemory(id=fixture.fixture_id, tenant_id=tenant, user_id=user, agent_id=agent, session_id="mpr2-retrieval-baseline", type=fixture.type, topic=fixture.topic, content=fixture.content, embedding=list(vector), content_hash=canonical_content_hash(fixture.content), source="manual", embedding_provider=batch.identity.provider, embedding_model=batch.identity.model, embedding_version=batch.identity.version, embedding_dimension=batch.identity.dimension, tags=list(fixture.tags), scope_service=fixture.scope_service, scope_env=fixture.scope_env))
            if result.status != "written":
                raise ValueError("v2 fixture write was not isolated and fresh")
            ids.append(fixture.fixture_id)
        return tuple(ids)

    async def execute_dev(self, *, top_k: int = 10, min_similarity: float = -1.0) -> tuple[list[V2Observation], list[tuple[str, str]]]:
        service = MemorySearchService(self.repository, self.embedding_service, top_k=top_k, min_similarity=min_similarity)
        observations: list[V2Observation] = []
        errors: list[tuple[str, str]] = []
        for case in (q for q in self.dataset.queries if q.split == "dev"):
            tenant, user, agent = V2_IDENTITY_MAP[case.identity]
            started = perf_counter()
            try:
                results = await service.search_memory(tenant, user, agent, case.query, case.optional_type, scope_service=case.scope_service, scope_env=case.scope_env, tags=case.tags)
                returned = tuple(V2ReturnedEvidence(rank=index, evidence_id=self._evidence_by_fixture[result.id], similarity=float(result.similarity)) for index, result in enumerate(results, 1))
                observations.append(_observation(case, returned, (perf_counter() - started) * 1000, self._fixtures))
            except Exception as exc:
                code = exc.code if isinstance(exc, MemoryPersistenceError) else "memory_retrieval_infrastructure_error"
                errors.append((case.case_id, code))
        return observations, errors


def _observation(case: V2Query, returned: tuple[V2ReturnedEvidence, ...], latency_ms: float, fixtures: dict[str, V2Fixture]) -> V2Observation:
    ids = tuple(item.evidence_id for item in returned)
    relevant = tuple(e for e in ids if e in case.relevant_evidence_ids)
    forbidden = tuple(e for e in ids if e in case.forbidden_evidence_ids)
    identity_leakage = sum(fixtures[e].identity != case.identity for e in ids)
    top1 = returned[0].similarity if returned else None
    top2 = returned[1].similarity if len(returned) > 1 else None
    margin = top1 - top2 if top1 is not None and top2 is not None else None
    passed = bool(relevant) and not forbidden and identity_leakage == 0 if case.track == "semantic_ranking" else not ids and identity_leakage == 0 and not forbidden
    return V2Observation(case_id=case.case_id, split=case.split, track=case.track, category=case.category, query_sha256=hashlib.sha256(case.query.encode()).hexdigest(), returned=returned, relevant_evidence_ids=tuple(case.relevant_evidence_ids), forbidden_evidence_ids=tuple(case.forbidden_evidence_ids), relevant_matches=relevant, forbidden_matches=forbidden, identity_leakage=identity_leakage, unexpected_result=bool(ids), query_latency_ms=latency_ms, top1_similarity=top1, top2_similarity=top2, top1_top2_margin=margin, passed=passed)


def summarize_v2_tracks(observations: list[V2Observation], *, top_k: int = 3) -> dict[str, Any]:
    semantic = [o for o in observations if o.track == "semantic_ranking"]
    no_match = [o for o in observations if o.track == "no_match"]
    isolation = [o for o in observations if o.track == "identity_isolation"]
    hits1 = sum(bool(set(o.relevant_matches) & {r.evidence_id for r in o.returned[:1]}) for o in semantic)
    hits3 = sum(bool(set(o.relevant_matches) & {r.evidence_id for r in o.returned[:3]}) for o in semantic)
    recall3 = sum(len(set(o.relevant_evidence_ids) & {r.evidence_id for r in o.returned[:3]}) / (len(o.relevant_evidence_ids) or 1) for o in semantic)
    recall5 = sum(len(set(o.relevant_evidence_ids) & {r.evidence_id for r in o.returned[:5]}) / (len(o.relevant_evidence_ids) or 1) for o in semantic)
    mrr = sum(next((1 / r.rank for r in o.returned if r.evidence_id in o.relevant_evidence_ids), 0.0) for o in semantic)
    return {
        "semantic_ranking": {
            "planned": 30, "executed": len(semantic), "hit_rate_at_1": hits1 / 30, "hit_rate_at_3": hits3 / 30,
            "recall_at_3": recall3 / 30, "recall_at_5": recall5 / 30, "mrr": mrr / 30,
            "hard_negative_forbidden_violations": sum(len(o.forbidden_matches) for o in semantic if o.category == "hard_negative"),
        },
        "no_match": {"planned": 10, "executed": len(no_match), "false_positive_count": sum(bool(o.returned) for o in no_match), "false_positive_rate": sum(bool(o.returned) for o in no_match) / 10},
        "identity_isolation": {"planned": 8, "executed": len(isolation), "identity_leakage": sum(o.identity_leakage for o in isolation), "forbidden_evidence_violations": sum(len(o.forbidden_matches) for o in isolation), "weak_related_unexpected_result_cases": sum(bool(o.returned) for o in isolation), "empty_result_pass_rate": sum(not o.returned and not o.forbidden_matches and o.identity_leakage == 0 for o in isolation) / 8},
    }


def _thresholds(values: list[float]) -> list[float]:
    unique = sorted({round(v, 9) for v in values if math.isfinite(v)})
    if not unique:
        return [0.0]
    candidates = {unique[0] - 1e-6, unique[-1] + 1e-6}
    for left, right in zip(unique, unique[1:]):
        candidates.add(round((left + right) / 2, 9))
    return sorted(candidates)


def calibrate_v2(observations: list[V2Observation], dataset: V2Dataset) -> dict[str, Any]:
    dev_cases = {q.case_id: q for q in dataset.queries if q.split == "dev"}
    top1_values = [o.top1_similarity for o in observations if o.top1_similarity is not None]
    margins = [o.top1_top2_margin for o in observations if o.top1_top2_margin is not None]
    points: list[CalibrationPoint] = []
    for threshold in _thresholds(top1_values):
        points.append(CalibrationPoint("top1", threshold, None, _gate_for(observations, dev_cases, threshold, None)))
    for threshold in _thresholds(top1_values):
        for margin in _thresholds(margins):
            points.append(CalibrationPoint("top1_margin", threshold, margin, _gate_for(observations, dev_cases, threshold, margin)))
    feasible = [p for p in points if p.gate.feasible]
    frontier = [p for p in points if not any(_dominates_gate(other.gate, p.gate) for other in points if other is not p)]
    status = "calibration_frontier_available" if feasible else "no_simple_gate_separates_positive_and_negative"
    return {"status": status, "candidate_count": len(points), "feasible_count": len(feasible), "pareto_frontier": [p.to_dict() for p in frontier], "points": [p.to_dict() for p in points], "production_candidate": None, "inputs": "top1 similarity and top1/top2 margin only; no case id, category or gold labels"}


def _gate_for(observations: list[V2Observation], cases: dict[str, V2Query], threshold: float, margin: float | None) -> GateResult:
    accepted, rejected, false_accept, false_reject = [], [], [], []
    for observation in observations:
        accept = observation.top1_similarity is not None and observation.top1_similarity >= threshold and (margin is None or (observation.top1_top2_margin is not None and observation.top1_top2_margin >= margin))
        if accept:
            accepted.append(observation.case_id)
        else:
            rejected.append(observation.case_id)
        if observation.track in {"no_match", "identity_isolation"} and accept:
            false_accept.append(observation.case_id)
        if observation.track == "semantic_ranking" and not accept:
            false_reject.append(observation.case_id)
    semantic = [o for o in observations if o.track == "semantic_ranking" and o.case_id in accepted]
    no_match = [o for o in observations if o.track == "no_match" and o.case_id in accepted]
    isolation = [o for o in observations if o.track == "identity_isolation"]
    accepted_isolation = [o for o in isolation if o.case_id in accepted]
    hard_negative_forbidden = sum(
        len(o.forbidden_matches)
        for o in observations
        if o.track == "semantic_ranking" and o.category == "hard_negative" and o.case_id in accepted
    )
    return GateResult(tuple(accepted), tuple(rejected), len(semantic) / 30, len(false_reject) / 30, len(no_match) / 10, sum(not o.returned or o.case_id not in accepted for o in isolation) / 8, sum(bool(set(o.relevant_evidence_ids) & {r.evidence_id for r in o.returned[:3]}) for o in semantic) / 30, sum(len(set(o.relevant_evidence_ids) & {r.evidence_id for r in o.returned[:3]}) / (len(o.relevant_evidence_ids) or 1) for o in semantic) / 30, sum(next((1 / r.rank for r in o.returned if r.evidence_id in o.relevant_evidence_ids), 0.0) for o in semantic) / 30, sum(o.identity_leakage for o in accepted_isolation), tuple(false_accept), tuple(false_reject), not false_accept and hard_negative_forbidden == 0 and sum(o.identity_leakage for o in accepted_isolation) == 0)


def _dominates_gate(left: GateResult, right: GateResult) -> bool:
    lv = (left.answerable_acceptance, left.semantic_recall_at_3, left.semantic_mrr, -left.no_answer_fpr, left.isolation_empty_rate, -left.identity_leakage)
    rv = (right.answerable_acceptance, right.semantic_recall_at_3, right.semantic_mrr, -right.no_answer_fpr, right.isolation_empty_rate, -right.identity_leakage)
    return all(a >= b for a, b in zip(lv, rv, strict=True)) and any(a > b for a, b in zip(lv, rv, strict=True))


def serialize_v2_artifact(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
