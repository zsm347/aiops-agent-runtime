from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Sequence

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.utils import get_tokenizer

from superbiz_agent.rag.document_loader import LoadedRagDocument, RagDocumentLoader
from superbiz_agent.rag.models import (
    RagChunkingConfig,
    RagDocumentContentType,
    RagPreparedChunk,
    RagSourceDocument,
)


_HEADER_PATH_SEPARATOR = "\x1f"
_MARKDOWN_HEADER_RE = re.compile(r"^(?P<marker>#+)\s(?P<title>.*)$")


@dataclass(frozen=True)
class _ChunkDraft:
    content: str
    heading_path: tuple[str, ...] = ()
    section_title: str | None = None


def normalize_chunk_content(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def _has_information(content: str) -> bool:
    return any(character.isalnum() for character in content)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    identity = json.dumps(
        [document_id, chunk_index, content],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _prepare_chunks(
    source: RagSourceDocument,
    drafts: Sequence[_ChunkDraft],
) -> tuple[RagPreparedChunk, ...]:
    prepared: list[RagPreparedChunk] = []
    for draft in drafts:
        content = normalize_chunk_content(draft.content)
        if not content or not _has_information(content):
            continue
        chunk_index = len(prepared)
        prepared.append(
            RagPreparedChunk(
                chunk_id=_chunk_id(source.document_id, chunk_index, content),
                tenant_id=source.tenant_id,
                knowledge_base_id=source.knowledge_base_id,
                document_id=source.document_id,
                document_name=source.document_name,
                source_uri=source.source_uri,
                heading_path=draft.heading_path,
                section_title=draft.section_title,
                page_start=source.page_start,
                page_end=source.page_end,
                page_metadata=dict(source.page_metadata),
                chunk_index=chunk_index,
                content_hash=_content_hash(content),
                content=content,
            )
        )
    return tuple(prepared)


class MarkdownChunkPostProcessor:
    """Turn MarkdownNodeParser sections into stable, citation-ready chunks."""

    def __init__(self, config: RagChunkingConfig | None = None) -> None:
        self.config = config or RagChunkingConfig()
        self._tokenizer = get_tokenizer()
        self._secondary_splitter = SentenceSplitter.from_defaults(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            tokenizer=self._tokenizer,
            include_metadata=False,
            include_prev_next_rel=False,
        )

    def process(
        self,
        source: RagSourceDocument,
        nodes: Sequence[BaseNode],
    ) -> tuple[RagPreparedChunk, ...]:
        drafts: list[_ChunkDraft] = []
        for node in nodes:
            section_text = normalize_chunk_content(node.get_content(metadata_mode="none"))
            parent_path = self._parent_heading_path(node)
            section_title, body = self._section_parts(section_text)
            heading_path = parent_path + ((section_title,) if section_title else ())

            if not body or not _has_information(body):
                continue

            heading_context = self._heading_context(heading_path)
            for body_chunk in self._split_body(body, heading_context):
                normalized_body = normalize_chunk_content(body_chunk)
                if not normalized_body or not _has_information(normalized_body):
                    continue
                content = self._with_heading_context(heading_context, normalized_body)
                token_count = len(self._tokenizer(content))
                if token_count > self.config.chunk_size:
                    raise ValueError(
                        "Final Markdown chunk exceeds chunk_size after adding heading context: "
                        f"{token_count} > {self.config.chunk_size}."
                    )
                drafts.append(
                    _ChunkDraft(
                        content=content,
                        heading_path=heading_path,
                        section_title=section_title,
                    )
                )
        return _prepare_chunks(source, drafts)

    def _split_body(self, body: str, heading_context: str) -> list[str]:
        if not heading_context:
            return self._secondary_splitter.split_text(body)

        heading_tokens = len(self._tokenizer(heading_context))
        body_budget = self.config.chunk_size - heading_tokens
        if body_budget <= 0:
            raise ValueError(
                "Markdown heading context exhausts chunk_size: "
                f"{heading_tokens} >= {self.config.chunk_size}."
            )
        splitter = SentenceSplitter.from_defaults(
            chunk_size=body_budget,
            chunk_overlap=min(self.config.chunk_overlap, max(0, body_budget - 1)),
            tokenizer=self._tokenizer,
            include_metadata=False,
            include_prev_next_rel=False,
        )
        return splitter.split_text(body)

    @staticmethod
    def _parent_heading_path(node: BaseNode) -> tuple[str, ...]:
        raw_path = str(node.metadata.get("header_path", ""))
        return tuple(
            part.strip()
            for part in raw_path.split(_HEADER_PATH_SEPARATOR)
            if part.strip()
        )

    @staticmethod
    def _section_parts(section_text: str) -> tuple[str | None, str]:
        first_line, separator, remainder = section_text.partition("\n")
        header_match = _MARKDOWN_HEADER_RE.match(first_line)
        if header_match is None:
            return None, section_text
        title = header_match.group("title").strip()
        return title or None, remainder.strip() if separator else ""

    @staticmethod
    def _heading_context(heading_path: tuple[str, ...]) -> str:
        if not heading_path:
            return ""
        return f"{' > '.join(heading_path)}\n\n"

    @staticmethod
    def _with_heading_context(heading_context: str, body: str) -> str:
        if not heading_context:
            return body
        return f"{heading_context}{body}"


class RagDocumentChunker:
    def __init__(
        self,
        config: RagChunkingConfig | None = None,
        loader: RagDocumentLoader | None = None,
    ) -> None:
        self.config = config or RagChunkingConfig()
        self.loader = loader or RagDocumentLoader()
        self._markdown_parser = MarkdownNodeParser.from_defaults(
            include_metadata=True,
            include_prev_next_rel=False,
            header_path_separator=_HEADER_PATH_SEPARATOR,
        )
        self._markdown_postprocessor = MarkdownChunkPostProcessor(self.config)
        self._text_splitter = SentenceSplitter.from_defaults(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            include_metadata=False,
            include_prev_next_rel=False,
        )

    def chunk(self, source: RagSourceDocument) -> tuple[RagPreparedChunk, ...]:
        loaded = self.loader.load(source)
        if loaded.content_type is RagDocumentContentType.MARKDOWN:
            return self._chunk_markdown(loaded)
        return self._chunk_text(loaded)

    def _chunk_markdown(self, loaded: LoadedRagDocument) -> tuple[RagPreparedChunk, ...]:
        if loaded.llama_document is None:
            raise ValueError("Markdown input must contain a LlamaIndex Document.")
        nodes = self._markdown_parser.get_nodes_from_documents([loaded.llama_document])
        return self._markdown_postprocessor.process(loaded.source, nodes)

    def _chunk_text(self, loaded: LoadedRagDocument) -> tuple[RagPreparedChunk, ...]:
        drafts = [
            _ChunkDraft(content=chunk)
            for chunk in self._text_splitter.split_text(loaded.content)
        ]
        return _prepare_chunks(loaded.source, drafts)


def chunk_document(
    source: RagSourceDocument,
    config: RagChunkingConfig | None = None,
) -> tuple[RagPreparedChunk, ...]:
    return RagDocumentChunker(config=config).chunk(source)
