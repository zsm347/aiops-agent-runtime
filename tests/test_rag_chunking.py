from __future__ import annotations

import hashlib

import pytest
from llama_index.core import Document
from llama_index.core.utils import get_tokenizer

from superbiz_agent.rag.chunking import RagDocumentChunker
from superbiz_agent.rag.document_loader import RagDocumentLoader
from superbiz_agent.rag.models import RagChunkingConfig, RagSourceDocument


def _source(
    content: str | bytes,
    *,
    content_type: str = "markdown",
    tenant_id: str = "tenant-a",
) -> RagSourceDocument:
    return RagSourceDocument(
        document_id="doc-pod-runbook",
        tenant_id=tenant_id,
        knowledge_base_id="kb-operations",
        document_name="pod-runbook.md",
        source_uri="runbooks/pod-runbook.md",
        content=content,
        content_type=content_type,
        page_start=3,
        page_end=4,
        page_metadata={"parser": "trusted-parser", "page_label": "3-4"},
    )


def test_markdown_loader_builds_llamaindex_document_with_trusted_metadata() -> None:
    loaded = RagDocumentLoader().load(_source(b"# Runbook\n\nBody"))

    assert isinstance(loaded.llama_document, Document)
    assert loaded.content == "# Runbook\n\nBody"
    assert loaded.llama_document.metadata["tenant_id"] == "tenant-a"
    assert loaded.llama_document.metadata["document_id"] == "doc-pod-runbook"


def test_multilevel_heading_path_and_heading_only_parents() -> None:
    chunks = RagDocumentChunker().chunk(
        _source(
            """# Pod 排障

## 容器启动失败

### CrashLoopBackOff

检查 describe pod 和容器日志。
"""
        )
    )

    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Pod 排障", "容器启动失败", "CrashLoopBackOff")
    assert chunks[0].section_title == "CrashLoopBackOff"
    assert chunks[0].content.startswith("Pod 排障 > 容器启动失败 > CrashLoopBackOff\n\n")


def test_sections_with_body_create_chunks_in_source_order() -> None:
    chunks = RagDocumentChunker().chunk(
        _source(
            """# Pod 排障

先确认 Pod 所属命名空间。

## 容器日志

执行 kubectl logs 查看启动错误。
"""
        )
    )

    assert [chunk.heading_path for chunk in chunks] == [
        ("Pod 排障",),
        ("Pod 排障", "容器日志"),
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_long_markdown_section_is_split_and_inherits_all_metadata() -> None:
    body = " ".join(f"Step {index} checks the container state." for index in range(80))
    source = _source(f"# Pod 排障\n\n## CrashLoopBackOff\n\n{body}")
    config = RagChunkingConfig(chunk_size=40, chunk_overlap=5)
    chunker = RagDocumentChunker(config)
    chunks = chunker.chunk(source)
    repeated = chunker.chunk(source)
    tokenizer = get_tokenizer()

    assert len(chunks) > 1
    assert chunks == repeated
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.tenant_id == source.tenant_id
        assert chunk.knowledge_base_id == source.knowledge_base_id
        assert chunk.document_id == source.document_id
        assert chunk.document_name == source.document_name
        assert chunk.source_uri == source.source_uri
        assert chunk.heading_path == ("Pod 排障", "CrashLoopBackOff")
        assert chunk.section_title == "CrashLoopBackOff"
        assert chunk.page_start == 3
        assert chunk.page_end == 4
        assert chunk.page_metadata == source.page_metadata
        assert len(tokenizer(chunk.content)) <= config.chunk_size
        assert chunk.content_hash == hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        assert chunk.chunk_id == repeated[index].chunk_id


def test_heading_path_keeps_slashes_inside_each_heading() -> None:
    chunks = RagDocumentChunker().chunk(
        _source(
            """# HTTP/API 排障

## 网关/超时

检查反向代理和上游超时配置。
"""
        )
    )

    assert len(chunks) == 1
    assert chunks[0].heading_path == ("HTTP/API 排障", "网关/超时")
    assert chunks[0].section_title == "网关/超时"
    assert chunks[0].content.startswith("HTTP/API 排障 > 网关/超时\n\n")


def test_heading_context_that_exhausts_budget_is_rejected() -> None:
    title = "HTTP API Gateway Production Troubleshooting Reference"

    with pytest.raises(ValueError, match="heading context exhausts chunk_size"):
        RagDocumentChunker(RagChunkingConfig(chunk_size=5, chunk_overlap=1)).chunk(
            _source(f"# {title}\n\nBody remains factual.")
        )


@pytest.mark.parametrize("content_type", ["txt", "parsed_text"])
def test_txt_and_parsed_text_use_plain_text_splitting(content_type: str) -> None:
    content = " ".join(f"Operational sentence {index}." for index in range(60))
    chunks = RagDocumentChunker(RagChunkingConfig(chunk_size=35, chunk_overlap=4)).chunk(
        _source(content, content_type=content_type)
    )

    assert len(chunks) > 1
    assert all(chunk.heading_path == () for chunk in chunks)
    assert all(chunk.section_title is None for chunk in chunks)


@pytest.mark.parametrize("content", ["", "  \n\t", b"\r\n  "])
def test_blank_content_is_rejected(content: str | bytes) -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        RagDocumentChunker().chunk(_source(content))


def test_unsupported_content_type_and_non_utf8_bytes_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported document content type"):
        RagDocumentChunker().chunk(_source("body", content_type="application/pdf"))
    with pytest.raises(ValueError, match="valid UTF-8"):
        RagDocumentChunker().chunk(_source(b"\xff\xfe", content_type="txt"))


def test_symbol_only_and_heading_only_sections_are_filtered() -> None:
    chunks = RagDocumentChunker().chunk(
        _source(
            """# Parent only

## Symbols only

--- *** !!!

## Useful

Run kubectl describe pod.
"""
        )
    )

    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Parent only", "Useful")


def test_chunk_ids_hashes_order_and_metadata_are_repeatable() -> None:
    source = _source(
        "# Runbook\r\n\r\nFirst factual paragraph.\r\n\r\n## Details\r\n\r\nSecond fact."
    )
    chunker = RagDocumentChunker(RagChunkingConfig(chunk_size=30, chunk_overlap=3))

    first = chunker.chunk(source)
    second = chunker.chunk(source)

    assert first == second
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))
    assert all("\r" not in chunk.content for chunk in first)
    assert all(
        chunk.content_hash == hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        for chunk in first
    )


def test_identical_content_for_two_tenants_keeps_backend_identity_separate() -> None:
    content = "# Shared title\n\nThe same factual body."
    tenant_a_chunks = RagDocumentChunker().chunk(_source(content, tenant_id="tenant-a"))
    tenant_b_chunks = RagDocumentChunker().chunk(_source(content, tenant_id="tenant-b"))

    assert [chunk.tenant_id for chunk in tenant_a_chunks] == ["tenant-a"]
    assert [chunk.tenant_id for chunk in tenant_b_chunks] == ["tenant-b"]
    assert tenant_a_chunks[0].chunk_id == tenant_b_chunks[0].chunk_id
    assert tenant_a_chunks[0].content_hash == tenant_b_chunks[0].content_hash
