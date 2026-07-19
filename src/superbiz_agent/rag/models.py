from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from numbers import Real
from typing import Any


_RAG_CITATION_METADATA_KEYS = frozenset(
    {
        "documentId",
        "documentName",
        "knowledgeBaseId",
        "headingPath",
        "sectionTitle",
        "pageStart",
        "pageEnd",
        "retrievalSource",
        "confidence",
        "evidenceType",
    }
)
_RAG_ID_MAX = 128
_RAG_CONTENT_MAX = 16000
_RAG_CITATION_TEXT_MAX = 512
_RAG_HEADING_DEPTH_MAX = 16
_RAG_TOOL_JSON_MAX = 50000


class RagToolResultContractError(ValueError):
    """Raised when a retrieval result cannot be safely exposed to a tool caller."""


class RagRetrievalMode(str, Enum):
    DENSE = "dense_only"
    BM25 = "bm25_only"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class RagRetrievalScope:
    tenant_id: str
    user_id: str
    agent_id: str
    run_id: str
    tool_call_id: str


@dataclass(frozen=True)
class RagRetrievalRequest:
    query: str
    scope: RagRetrievalScope


@dataclass(frozen=True)
class RagChunk:
    id: str
    source: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tool_result(self, ref: int) -> dict[str, Any]:
        _validate_nonblank_string(self.id, "chunk id", _RAG_ID_MAX)
        _validate_nonblank_string(self.source, "chunk source", _RAG_CITATION_TEXT_MAX)
        _validate_nonblank_string(self.content, "chunk content", _RAG_CONTENT_MAX)
        if isinstance(ref, bool) or not isinstance(ref, int) or ref < 1:
            raise RagToolResultContractError("Chunk reference is invalid.")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, Real)
            or not math.isfinite(float(self.score))
        ):
            raise RagToolResultContractError("Chunk score is invalid.")
        if not isinstance(self.metadata, Mapping):
            raise RagToolResultContractError("Chunk metadata is invalid.")

        try:
            result = _validated_citation_metadata(self.metadata)
        except RagToolResultContractError:
            raise
        except Exception:
            raise RagToolResultContractError("Chunk metadata is invalid.") from None
        result.update({
            "id": self.id,
            "ref": ref,
            "source": self.source,
            "content": self.content,
        })
        if self.score is not None:
            result["score"] = self.score
        return result


@dataclass(frozen=True)
class RagRetrievalResult:
    chunks: tuple[RagChunk, ...] = ()
    message: str | None = None

    def to_tool_result(self) -> dict[str, Any]:
        if not isinstance(self.chunks, tuple) or any(
            not isinstance(chunk, RagChunk) for chunk in self.chunks
        ):
            raise RagToolResultContractError("Retrieval chunks are invalid.")
        if not self.chunks:
            message = self.message or "No relevant documents found in the knowledge base."
            _validate_nonblank_string(message, "retrieval message", 2000)
            result = {
                "status": "no_results",
                "message": message,
            }
        else:
            result = {
                "status": "ok",
                "count": len(self.chunks),
                "chunks": [
                    chunk.to_tool_result(index) for index, chunk in enumerate(self.chunks, 1)
                ],
            }
        try:
            serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            raise RagToolResultContractError("Retrieval output is not valid JSON.") from None
        if len(serialized) > _RAG_TOOL_JSON_MAX:
            raise RagToolResultContractError("Retrieval output exceeds the size limit.")
        return result


def _validated_citation_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: metadata[key] for key in _RAG_CITATION_METADATA_KEYS if key in metadata}
    for key in ("documentId", "knowledgeBaseId"):
        if key in result:
            _validate_nonblank_string(result[key], key, _RAG_ID_MAX)
    for key in ("documentName", "sectionTitle"):
        if key in result and result[key] is not None:
            _validate_nonblank_string(result[key], key, _RAG_CITATION_TEXT_MAX)
    if "headingPath" in result:
        heading_path = result["headingPath"]
        if (
            not isinstance(heading_path, (list, tuple))
            or len(heading_path) > _RAG_HEADING_DEPTH_MAX
            or any(
                not isinstance(item, str)
                or not item.strip()
                or len(item) > _RAG_CITATION_TEXT_MAX
                for item in heading_path
            )
        ):
            raise RagToolResultContractError("headingPath metadata is invalid.")
        result["headingPath"] = list(heading_path)
    for key in ("pageStart", "pageEnd"):
        value = result.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise RagToolResultContractError(f"{key} metadata is invalid.")
    page_start = result.get("pageStart")
    page_end = result.get("pageEnd")
    if page_start is not None and page_end is not None and page_end < page_start:
        raise RagToolResultContractError("Citation page range is invalid.")
    expected_literals = {
        "retrievalSource": "hybrid",
        "confidence": "unavailable",
        "evidenceType": "untrusted_retrieved_document",
    }
    for key, expected in expected_literals.items():
        if key in result and result[key] != expected:
            raise RagToolResultContractError(f"{key} metadata is invalid.")
    return result


def _validate_nonblank_string(value: Any, name: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise RagToolResultContractError(f"{name} is invalid.")


class RagDocumentContentType(str, Enum):
    MARKDOWN = "markdown"
    TXT = "txt"
    PARSED_TEXT = "parsed_text"


class RagIngestionStatus(str, Enum):
    INDEXED = "indexed"
    DUPLICATE_SKIPPED = "duplicate_skipped"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True)
class RagIngestionRequest:
    tenant_id: str
    knowledge_base_id: str
    document_name: str
    source_uri: str
    content: str | bytes
    content_type: RagDocumentContentType | str
    created_by: str
    parser: str
    parser_version: str
    page_start: int | None = None
    page_end: int | None = None
    page_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_fields = {
            "tenant_id": self.tenant_id,
            "knowledge_base_id": self.knowledge_base_id,
            "document_name": self.document_name,
            "source_uri": self.source_uri,
            "created_by": self.created_by,
            "parser": self.parser,
            "parser_version": self.parser_version,
        }
        for field_name, value in required_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be blank.")
        if not isinstance(self.content, (str, bytes)):
            raise TypeError("content must be a string or bytes.")
        if not isinstance(self.page_metadata, Mapping):
            raise TypeError("page_metadata must be a mapping.")
        if self.page_start is not None and self.page_start < 1:
            raise ValueError("page_start must be positive when provided.")
        if self.page_end is not None and self.page_end < 1:
            raise ValueError("page_end must be positive when provided.")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end cannot be smaller than page_start.")


@dataclass(frozen=True)
class RagIngestionResult:
    status: RagIngestionStatus
    document_id: str
    content_hash: str
    chunk_count: int | None = None


@dataclass(frozen=True)
class RagSourceDocument:
    document_id: str
    tenant_id: str
    knowledge_base_id: str
    document_name: str
    source_uri: str
    content: str | bytes
    content_type: RagDocumentContentType | str
    page_start: int | None = None
    page_end: int | None = None
    page_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identity_fields = {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "knowledge_base_id": self.knowledge_base_id,
            "document_name": self.document_name,
            "source_uri": self.source_uri,
        }
        for field_name, value in identity_fields.items():
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank.")
        if self.page_start is not None and self.page_start < 1:
            raise ValueError("page_start must be positive when provided.")
        if self.page_end is not None and self.page_end < 1:
            raise ValueError("page_end must be positive when provided.")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end cannot be smaller than page_start.")


@dataclass(frozen=True)
class RagChunkingConfig:
    chunk_size: int = 1024
    chunk_overlap: int = 100

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")


@dataclass(frozen=True)
class RagPreparedChunk:
    chunk_id: str
    tenant_id: str
    knowledge_base_id: str
    document_id: str
    document_name: str
    source_uri: str
    heading_path: tuple[str, ...]
    section_title: str | None
    page_start: int | None
    page_end: int | None
    page_metadata: dict[str, Any]
    chunk_index: int
    content_hash: str
    content: str
