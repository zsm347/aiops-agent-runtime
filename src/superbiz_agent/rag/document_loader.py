from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from llama_index.core import Document

from superbiz_agent.rag.models import RagDocumentContentType, RagSourceDocument


_CONTENT_TYPE_ALIASES = {
    "markdown": RagDocumentContentType.MARKDOWN,
    "md": RagDocumentContentType.MARKDOWN,
    "text/markdown": RagDocumentContentType.MARKDOWN,
    "txt": RagDocumentContentType.TXT,
    "text": RagDocumentContentType.TXT,
    "text/plain": RagDocumentContentType.TXT,
    "parsed_text": RagDocumentContentType.PARSED_TEXT,
    "parsed-text": RagDocumentContentType.PARSED_TEXT,
}


@dataclass(frozen=True)
class LoadedRagDocument:
    source: RagSourceDocument
    content: str
    content_type: RagDocumentContentType
    llama_document: Document | None = None


class ExternalDocumentParserAdapter(Protocol):
    """Boundary for a future trusted parser that returns supported source content."""

    def parse(self, source: RagSourceDocument) -> RagSourceDocument: ...


class RagDocumentLoader:
    """Load trusted backend content without performing file or network access."""

    def load(self, source: RagSourceDocument) -> LoadedRagDocument:
        content_type = self._resolve_content_type(source.content_type)
        content = self._decode_utf8(source.content)
        if not content.strip():
            raise ValueError("document content must not be blank.")

        llama_document = None
        if content_type is RagDocumentContentType.MARKDOWN:
            llama_document = Document(
                text=content,
                id_=source.document_id,
                metadata={
                    "tenant_id": source.tenant_id,
                    "knowledge_base_id": source.knowledge_base_id,
                    "document_id": source.document_id,
                    "document_name": source.document_name,
                    "source_uri": source.source_uri,
                    "page_start": source.page_start,
                    "page_end": source.page_end,
                    "page_metadata": dict(source.page_metadata),
                },
            )

        return LoadedRagDocument(
            source=source,
            content=content,
            content_type=content_type,
            llama_document=llama_document,
        )

    @staticmethod
    def _decode_utf8(content: str | bytes) -> str:
        if isinstance(content, str):
            return content
        try:
            return content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("document content must be valid UTF-8.") from exc

    @staticmethod
    def _resolve_content_type(
        content_type: RagDocumentContentType | str,
    ) -> RagDocumentContentType:
        if isinstance(content_type, RagDocumentContentType):
            return content_type
        resolved = _CONTENT_TYPE_ALIASES.get(content_type.strip().lower())
        if resolved is None:
            raise ValueError(f"unsupported document content type: {content_type!r}.")
        return resolved
