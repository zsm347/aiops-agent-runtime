"""RAG package."""
from typing import TYPE_CHECKING, Any

from superbiz_agent.rag.models import (
    RagChunk,
    RagChunkingConfig,
    RagDocumentContentType,
    RagIngestionRequest,
    RagIngestionResult,
    RagIngestionStatus,
    RagPreparedChunk,
    RagRetrievalRequest,
    RagRetrievalResult,
    RagRetrievalScope,
    RagSourceDocument,
    RagToolResultContractError,
)
from superbiz_agent.rag.retrieval import (
    DefaultKnowledgeBaseResolver,
    FixtureRagRetrievalService,
    MilvusRagRetrievalService,
    RagKnowledgeBaseNotConfiguredError,
    RagRetrievalContractError,
    RagRetrievalDisabledError,
    RagRetrievalIsolationError,
    RagRetrievalService,
    RagRetrievalUnavailableError,
    RepositoryDefaultKnowledgeBaseResolver,
    UnavailableRagRetrievalService,
)

if TYPE_CHECKING:
    from superbiz_agent.rag.chunking import MarkdownChunkPostProcessor, RagDocumentChunker
    from superbiz_agent.rag.document_loader import (
        ExternalDocumentParserAdapter,
        LoadedRagDocument,
        RagDocumentLoader,
    )


_CHUNKING_EXPORTS = {
    "MarkdownChunkPostProcessor",
    "RagDocumentChunker",
    "chunk_document",
    "normalize_chunk_content",
}
_LOADER_EXPORTS = {
    "ExternalDocumentParserAdapter",
    "LoadedRagDocument",
    "RagDocumentLoader",
}
_INGESTION_EXPORTS = {
    "RagIngestionError",
    "RagIngestionService",
    "canonicalize_document_content",
    "document_content_hash",
}
_RUNTIME_EXPORTS = {"RealRagRuntime", "build_real_rag_retrieval_service"}


def __getattr__(name: str) -> Any:
    if name in _CHUNKING_EXPORTS:
        from superbiz_agent.rag import chunking

        return getattr(chunking, name)
    if name in _LOADER_EXPORTS:
        from superbiz_agent.rag import document_loader

        return getattr(document_loader, name)
    if name in _INGESTION_EXPORTS:
        from superbiz_agent.rag import ingestion

        return getattr(ingestion, name)
    if name in _RUNTIME_EXPORTS:
        from superbiz_agent.rag import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ExternalDocumentParserAdapter",
    "DefaultKnowledgeBaseResolver",
    "FixtureRagRetrievalService",
    "LoadedRagDocument",
    "MarkdownChunkPostProcessor",
    "MilvusRagRetrievalService",
    "RagChunk",
    "RagChunkingConfig",
    "RagDocumentChunker",
    "RagDocumentContentType",
    "RagDocumentLoader",
    "RagIngestionError",
    "RagIngestionRequest",
    "RagIngestionResult",
    "RagIngestionService",
    "RagIngestionStatus",
    "RagPreparedChunk",
    "RagKnowledgeBaseNotConfiguredError",
    "RagRetrievalContractError",
    "RagRetrievalDisabledError",
    "RagRetrievalIsolationError",
    "RagRetrievalRequest",
    "RagRetrievalResult",
    "RagRetrievalScope",
    "RagRetrievalService",
    "RagRetrievalUnavailableError",
    "RagSourceDocument",
    "RagToolResultContractError",
    "UnavailableRagRetrievalService",
    "RealRagRuntime",
    "RepositoryDefaultKnowledgeBaseResolver",
    "build_real_rag_retrieval_service",
    "chunk_document",
    "canonicalize_document_content",
    "document_content_hash",
    "normalize_chunk_content",
]
