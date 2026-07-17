from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from superbiz_agent.rag.models import RagChunk, RagPreparedChunk


RAG_F0_TENANT_ID = "rag-f0-tenant"
RAG_F0_KNOWLEDGE_BASE_ID = "rag-f0-default-kb"
RAG_F0_DATA_FILES = ("corpus.jsonl", "evidence.jsonl", "queries.jsonl", "qrels.jsonl")
_WHITESPACE_RE = re.compile(r"\s+")


class RagF0ContractError(ValueError):
    """Raised when the F0 dataset or retrieval artifact violates its frozen contract."""


class RagF0Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RagF0CorpusDocument(RagF0Model):
    document_id: str = Field(min_length=1, max_length=128)
    document_name: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=100)
    tenant_id: str
    knowledge_base_id: str
    source_url: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: Literal["CC-BY-4.0", "Apache-2.0"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RagF0Evidence(RagF0Model):
    evidence_id: str = Field(pattern=r"^ev-[0-9a-f]{20}$")
    document_id: str = Field(min_length=1, max_length=128)
    heading_path: tuple[str, ...] = Field(max_length=16)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    text_anchor: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_span(self) -> "RagF0Evidence":
        if self.line_end < self.line_start:
            raise ValueError("line_end cannot be smaller than line_start")
        return self


class RagF0Generation(RagF0Model):
    method: Literal["human_authored"]
    model: None = None
    batch: str = Field(min_length=1, max_length=128)


class RagF0Query(RagF0Model):
    query_id: str = Field(pattern=r"^q-(dev|holdout)-[0-9]{3}$")
    text: str = Field(min_length=5, max_length=2000)
    split: Literal["dev", "holdout"]
    category: Literal[
        "exact_keyword",
        "natural_language",
        "error_code",
        "configuration",
        "symptom_only",
        "multi_evidence",
        "hard_negative",
        "no_answer",
        "command",
    ]
    slices: tuple[str, ...] = Field(min_length=1)
    allow_no_answer: bool
    rationale: str = Field(min_length=10)
    generation: RagF0Generation


class RagF0Qrel(RagF0Model):
    query_id: str
    evidence_id: str
    relevance: int = Field(ge=1, le=3)
    rationale: str = Field(min_length=10)
    source_quote: str = Field(min_length=8)


class RagF0Manifest(RagF0Model):
    dataset_version: str
    status: Literal["pending_independent_dataset_acceptance"]
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: dict[str, str]
    fixed_scope: dict[str, str]
    default_split: Literal["dev"]
    holdout_policy: Literal["frozen_not_run_in_f0"]
    out_of_scope: tuple[str, ...]
    sources: tuple[dict[str, Any], ...]
    query_generation: dict[str, Any]


class RagF0Dataset(RagF0Model):
    root: Path
    manifest: RagF0Manifest
    corpus: tuple[RagF0CorpusDocument, ...]
    evidence: tuple[RagF0Evidence, ...]
    queries: tuple[RagF0Query, ...]
    qrels: tuple[RagF0Qrel, ...]

    @model_validator(mode="after")
    def validate_catalog(self) -> "RagF0Dataset":
        _require_unique("document_id", [row.document_id for row in self.corpus])
        _require_unique("document_name", [row.document_name for row in self.corpus])
        _require_unique("evidence_id", [row.evidence_id for row in self.evidence])
        _require_unique("query_id", [row.query_id for row in self.queries])
        _require_unique(
            "query/evidence qrel",
            [f"{row.query_id}\0{row.evidence_id}" for row in self.qrels],
        )

        documents = {row.document_id: row for row in self.corpus}
        evidence = {row.evidence_id: row for row in self.evidence}
        queries = {row.query_id: row for row in self.queries}
        qrel_counts = Counter(row.query_id for row in self.qrels)
        for row in self.corpus:
            if (
                row.tenant_id != RAG_F0_TENANT_ID
                or row.knowledge_base_id != RAG_F0_KNOWLEDGE_BASE_ID
            ):
                raise ValueError("corpus scope is not the frozen F0 tenant and KB")
            if hashlib.sha256(row.text.encode("utf-8")).hexdigest() != row.sha256:
                raise ValueError(f"corpus text hash mismatch for {row.document_id}")
        for row in self.evidence:
            document = documents.get(row.document_id)
            if document is None:
                raise ValueError(f"evidence references unknown document: {row.evidence_id}")
            if _normalize(row.text_anchor) not in _normalize(row.text):
                raise ValueError(f"evidence anchor is missing: {row.evidence_id}")
            source_lines = document.text.splitlines()
            if row.line_end > len(source_lines):
                raise ValueError(f"evidence source span is outside document: {row.evidence_id}")
        for row in self.qrels:
            query = queries.get(row.query_id)
            selected = evidence.get(row.evidence_id)
            if query is None or selected is None:
                raise ValueError("qrel references an unknown query or evidence")
            if query.allow_no_answer:
                raise ValueError(f"no-answer query has a qrel: {query.query_id}")
            if _normalize(row.source_quote) not in _normalize(selected.text):
                raise ValueError(f"qrel source quote is missing: {query.query_id}")
        for query in self.queries:
            if query.allow_no_answer != (qrel_counts[query.query_id] == 0):
                raise ValueError(f"query answerability/qrel mismatch: {query.query_id}")
            expected_prefix = f"q-{query.split}-"
            if not query.query_id.startswith(expected_prefix):
                raise ValueError(f"query split/id mismatch: {query.query_id}")
        if Counter(query.split for query in self.queries) != {"dev": 48, "holdout": 12}:
            raise ValueError("F0 split must remain 48 dev / 12 holdout")
        return self

    def queries_for_split(self, split: Literal["dev", "holdout"] = "dev") -> tuple[RagF0Query, ...]:
        return tuple(query for query in self.queries if query.split == split)

    def relevant_evidence_ids(self, query_id: str) -> tuple[str, ...]:
        return tuple(
            row.evidence_id
            for row in sorted(
                (item for item in self.qrels if item.query_id == query_id),
                key=lambda item: (-item.relevance, item.evidence_id),
            )
        )


class RagF0EvidenceMapper:
    def __init__(self, dataset: RagF0Dataset) -> None:
        self._document_by_name = {
            document.document_name: document.document_id for document in dataset.corpus
        }
        self._by_document_path = {
            (row.document_id, row.heading_path): row for row in dataset.evidence
        }
        self._by_document: dict[str, list[RagF0Evidence]] = {}
        for row in dataset.evidence:
            self._by_document.setdefault(row.document_id, []).append(row)

    def map_chunk(self, chunk: RagChunk) -> RagF0Evidence:
        raw_document_name = chunk.metadata.get("documentName", chunk.source)
        raw_heading_path = chunk.metadata.get("headingPath", ())
        if not isinstance(raw_document_name, str) or not isinstance(
            raw_heading_path, (list, tuple)
        ):
            raise RagF0ContractError("retrieved chunk citation metadata is malformed")
        document_id = self._document_by_name.get(raw_document_name)
        if document_id is None:
            raise RagF0ContractError(
                f"retrieved document is outside the F0 corpus: {raw_document_name}"
            )
        heading_path = tuple(raw_heading_path)
        selected = self._by_document_path.get((document_id, heading_path))
        if selected is not None:
            return selected

        normalized_content = _normalize(chunk.content)
        candidates = [
            row
            for row in self._by_document.get(document_id, [])
            if _normalize(row.text_anchor) in normalized_content
        ]
        if len(candidates) != 1:
            raise RagF0ContractError(
                f"retrieved chunk cannot be mapped uniquely to evidence: {chunk.id}"
            )
        return candidates[0]

    def map_prepared_chunk(self, chunk: RagPreparedChunk, *, document_name: str) -> RagF0Evidence:
        return self.map_chunk(
            RagChunk(
                id=chunk.chunk_id,
                source=document_name,
                content=chunk.content,
                metadata={
                    "documentName": document_name,
                    "headingPath": list(chunk.heading_path),
                },
            )
        )


def default_rag_f0_dataset_path(project_root: Path | None = None) -> Path:
    root = project_root or Path.cwd()
    return root / "evals" / "datasets" / "rag_f0"


def load_rag_f0_dataset(path: str | Path) -> RagF0Dataset:
    root = Path(path)
    manifest = RagF0Manifest.model_validate_json((root / "manifest.json").read_text("utf-8"))
    for filename in RAG_F0_DATA_FILES:
        digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if manifest.files.get(filename) != digest:
            raise RagF0ContractError(f"dataset file hash mismatch: {filename}")
    if manifest.dataset_sha256 != stable_rag_f0_dataset_hash(manifest.files):
        raise RagF0ContractError("dataset manifest hash mismatch")
    try:
        return RagF0Dataset(
            root=root,
            manifest=manifest,
            corpus=tuple(
                RagF0CorpusDocument.model_validate(row) for row in _jsonl(root / "corpus.jsonl")
            ),
            evidence=tuple(
                RagF0Evidence.model_validate(row) for row in _jsonl(root / "evidence.jsonl")
            ),
            queries=tuple(RagF0Query.model_validate(row) for row in _jsonl(root / "queries.jsonl")),
            qrels=tuple(RagF0Qrel.model_validate(row) for row in _jsonl(root / "qrels.jsonl")),
        )
    except RagF0ContractError:
        raise
    except Exception as exc:
        raise RagF0ContractError("RAG F0 dataset contract validation failed") from exc


def stable_rag_f0_dataset_hash(files: dict[str, str]) -> str:
    payload = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RagF0ContractError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise RagF0ContractError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _normalize(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip().casefold()


def _require_unique(name: str, values: list[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name}")
