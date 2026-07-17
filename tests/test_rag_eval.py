from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from superbiz_agent.config import Settings
from superbiz_agent.evals.rag_cases import (
    RAG_F0_KNOWLEDGE_BASE_ID,
    RAG_F0_TENANT_ID,
    RagF0ContractError,
    RagF0EvidenceMapper,
    default_rag_f0_dataset_path,
    load_rag_f0_dataset,
)
from superbiz_agent.evals.rag_runner import RagF0Runner, _query_metrics
from superbiz_agent.rag.chunking import RagDocumentChunker
from superbiz_agent.rag.embedding import RagQueryEmbedding
from superbiz_agent.rag.milvus_store import RagHybridSearchHit
from superbiz_agent.rag.models import (
    RagChunk,
    RagDocumentContentType,
    RagRetrievalResult,
    RagSourceDocument,
)
from superbiz_agent.rag.retrieval import MilvusRagRetrievalService


@pytest.fixture(scope="module")
def dataset():
    return load_rag_f0_dataset(default_rag_f0_dataset_path())


def test_dataset_contract_hash_sources_splits_and_gold(dataset) -> None:
    assert len(dataset.corpus) == 18
    assert len(dataset.evidence) == 140
    assert len(dataset.queries) == 60
    assert len(dataset.qrels) == 62
    assert len(dataset.queries_for_split("dev")) == 48
    assert len(dataset.queries_for_split("holdout")) == 12
    assert dataset.manifest.status == "pending_independent_dataset_acceptance"
    assert dataset.manifest.dataset_sha256 == (
        "0627e66b0b5b40cb1fbe326f2dfb980be2f43440d264056b5ebc774e157d26ec"
    )
    assert {row.license for row in dataset.corpus} == {"CC-BY-4.0", "Apache-2.0"}
    assert sum(query.allow_no_answer for query in dataset.queries) == 4
    assert sum("multi_evidence" in query.slices for query in dataset.queries) == 6
    assert all(query.generation.model is None for query in dataset.queries)
    assert all(row.rationale and row.source_quote for row in dataset.qrels)


def test_dataset_hash_tampering_fails_closed(dataset, tmp_path: Path) -> None:
    copied = tmp_path / "rag_f0"
    shutil.copytree(dataset.root, copied)
    with (copied / "queries.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RagF0ContractError, match="hash mismatch"):
        load_rag_f0_dataset(copied)


def test_every_production_chunk_maps_to_stable_evidence_and_back(dataset) -> None:
    mapper = RagF0EvidenceMapper(dataset)
    chunker = RagDocumentChunker()
    mapped: set[str] = set()
    chunk_count = 0
    for document in dataset.corpus:
        chunks = chunker.chunk(
            RagSourceDocument(
                document_id=document.document_id,
                tenant_id=document.tenant_id,
                knowledge_base_id=document.knowledge_base_id,
                document_name=document.document_name,
                source_uri=document.source_url,
                content=document.text,
                content_type=RagDocumentContentType.MARKDOWN,
            )
        )
        for chunk in chunks:
            mapped.add(
                mapper.map_prepared_chunk(chunk, document_name=document.document_name).evidence_id
            )
            chunk_count += 1
    assert chunk_count == 143
    assert mapped == {row.evidence_id for row in dataset.evidence}


def test_llamaindex_metrics_match_known_fixture_and_no_answer_rule() -> None:
    metrics = _query_metrics(
        expected_ids=("evidence-a", "evidence-b"),
        retrieved_ids=("irrelevant", "evidence-b", "evidence-a"),
        allow_no_answer=False,
    )
    ideal_dcg = 1 + 1 / math.log2(3)
    actual_dcg = 1 / math.log2(3) + 1 / math.log2(4)
    assert metrics == pytest.approx(
        {
            "recall_at_10": 1.0,
            "hit_rate_at_3": 1.0,
            "mrr_at_10": 0.5,
            "ndcg_at_3": actual_dcg / ideal_dcg,
            "ndcg_at_10": actual_dcg / ideal_dcg,
            "precision_at_3": 2 / 3,
            "average_precision_at_10": ((1 / 2) + (2 / 3)) / 2,
        }
    )
    assert _query_metrics(expected_ids=(), retrieved_ids=(), allow_no_answer=True) == {
        "no_answer_false_positive_rate": 0.0
    }
    assert _query_metrics(expected_ids=(), retrieved_ids=("unexpected",), allow_no_answer=True) == {
        "no_answer_false_positive_rate": 1.0
    }


class _GoldRetrievalService:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.calls = []
        self.evidence = {row.evidence_id: row for row in dataset.evidence}
        self.documents = {row.document_id: row for row in dataset.corpus}

    async def search(self, request):
        self.calls.append(request)
        query_id = request.scope.tool_call_id.removeprefix("rag-f0-")
        expected = self.dataset.relevant_evidence_ids(query_id)
        chunks = []
        for rank, evidence_id in enumerate(expected, 1):
            evidence = self.evidence[evidence_id]
            document = self.documents[evidence.document_id]
            chunks.append(
                RagChunk(
                    id=f"chunk-{query_id}-{rank}",
                    source=document.document_name,
                    content=evidence.text,
                    score=1 / rank,
                    metadata={
                        "documentName": document.document_name,
                        "headingPath": list(evidence.heading_path),
                    },
                )
            )
        return RagRetrievalResult(chunks=tuple(chunks))


@pytest.mark.asyncio
async def test_runner_calls_service_with_backend_scope_and_is_reproducible(dataset) -> None:
    service = _GoldRetrievalService(dataset)
    first = await RagF0Runner(
        retrieval_service=service,
        dataset=dataset,
        bootstrap_samples=100,
    ).run_dev()
    second = await RagF0Runner(
        retrieval_service=service,
        dataset=dataset,
        bootstrap_samples=100,
    ).run_dev()

    assert first.status == "completed"
    assert first.query_count == 48
    assert first.infrastructure_failure_count == 0
    assert first.metrics == second.metrics
    assert first.slice_metrics == second.slice_metrics
    assert first.metrics["recall_at_10"].value == 1.0
    assert first.metrics["hit_rate_at_3"].value == 1.0
    assert first.metrics["no_answer_false_positive_rate"].value == 0.0
    assert all(call.scope.tenant_id == RAG_F0_TENANT_ID for call in service.calls)
    assert all("knowledge_base_id" not in call.query for call in service.calls)
    assert first.ablations["dense_only"].startswith("not_available")
    assert first.out_of_scope["inactive_documents"].startswith("out_of_scope")


@pytest.mark.asyncio
async def test_runner_mapping_failure_is_contract_error(dataset) -> None:
    class BadService:
        async def search(self, request):
            del request
            return RagRetrievalResult(
                chunks=(
                    RagChunk(
                        id="outside",
                        source="outside.md",
                        content="outside corpus",
                        metadata={"documentName": "outside.md", "headingPath": []},
                    ),
                )
            )

    with pytest.raises(RagF0ContractError, match="outside the F0 corpus"):
        await RagF0Runner(
            retrieval_service=BadService(),
            dataset=dataset,
            bootstrap_samples=100,
        ).run_dev()


class _Resolver:
    async def resolve(self, tenant_id: str) -> str:
        assert tenant_id == RAG_F0_TENANT_ID
        return RAG_F0_KNOWLEDGE_BASE_ID


class _Embedding:
    async def embed_query(self, text: str) -> RagQueryEmbedding:
        assert text
        return RagQueryEmbedding((1.0, 0.0), "test", "test", "v1", 2)


class _Documents:
    async def get_document_statuses(self, **kwargs):
        return {document_id: "active" for document_id in kwargs["document_ids"]}


class _Store:
    def __init__(self) -> None:
        self.hits = tuple(
            RagHybridSearchHit(
                chunk_id=f"chunk-{index}",
                document_id=f"document-{index}",
                tenant_id=RAG_F0_TENANT_ID,
                knowledge_base_id=RAG_F0_KNOWLEDGE_BASE_ID,
                document_name=f"document-{index}.md",
                content=f"evidence {index}",
                score=1 / index,
                heading_path=("Runbook",),
                section_title="Runbook",
                page_start=None,
                page_end=None,
            )
            for index in range(1, 11)
        )

    async def hybrid_search(self, **kwargs):
        return self.hits[: kwargs["similarity_top_k"]]


@pytest.mark.asyncio
async def test_eval_top10_top3_prefix_matches_default_service_contract() -> None:
    common = dict(
        _env_file=None,
        rag_fixture_mode=False,
        rag_embedding_provider="test",
        rag_embedding_model="test",
        rag_embedding_version="v1",
        rag_embedding_dimension=2,
        rag_hybrid_top_k=10,
    )
    dependencies = dict(
        knowledge_base_resolver=_Resolver(),
        document_repository=_Documents(),
        embedding_service=_Embedding(),
        chunk_store=_Store(),
    )
    top10 = await MilvusRagRetrievalService(
        settings=Settings(**common, rag_final_top_k=10), **dependencies
    ).search(_request())
    top3 = await MilvusRagRetrievalService(
        settings=Settings(**common, rag_final_top_k=3), **dependencies
    ).search(_request())
    assert [chunk.id for chunk in top3.chunks] == [chunk.id for chunk in top10.chunks[:3]]


def _request():
    from superbiz_agent.rag.models import RagRetrievalRequest, RagRetrievalScope

    return RagRetrievalRequest(
        query="prefix check",
        scope=RagRetrievalScope(
            tenant_id=RAG_F0_TENANT_ID,
            user_id="eval",
            agent_id="eval",
            run_id="eval",
            tool_call_id="eval",
        ),
    )
