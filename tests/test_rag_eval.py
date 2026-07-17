from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

import scripts.run_rag_f0_baseline as baseline_entrypoint
from scripts.run_rag_f0_baseline import (
    ALEMBIC_HEAD,
    RagF0SafetyError,
    _cleanup_postgres,
    _drop_owned_collection,
    _static_preflight,
    _validate_milvus_preflight,
    _validate_postgres_preflight,
)

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


def test_default_dataset_path_does_not_depend_on_cwd(
    dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_rag_f0_dataset_path() == dataset.root


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
            "binary_ndcg_at_3": actual_dcg / ideal_dcg,
            "binary_ndcg_at_10": actual_dcg / ideal_dcg,
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
async def test_runner_with_partial_infrastructure_failure_has_no_aggregate_metrics(dataset) -> None:
    class PartiallyFailingService(_GoldRetrievalService):
        async def search(self, request):
            if request.scope.tool_call_id == "rag-f0-q-dev-001":
                raise TimeoutError
            return await super().search(request)

    report = await RagF0Runner(
        retrieval_service=PartiallyFailingService(dataset),
        dataset=dataset,
        bootstrap_samples=100,
    ).run_dev()

    assert report.status == "infrastructure_pending"
    assert report.infrastructure_failure_count == 1
    assert report.metrics == {}
    assert report.slice_metrics == {}
    assert report.latency_ms == {"p50": None, "p95": None}


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


def test_real_entrypoint_static_preflight_requires_exact_scoped_resources() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://localhost/superbiz_rag_f0_dev",
        rag_enabled=True,
        rag_fixture_mode=False,
        rag_rerank_enabled=False,
        rag_embedding_api_key="test-only",
        rag_milvus_collection="rag_f0_dev",
    )
    assert (
        _static_preflight(
            settings,
            confirmed=True,
            expected_database_name="superbiz_rag_f0_dev",
            expected_milvus_collection="rag_f0_dev",
        )
        is None
    )
    assert (
        _static_preflight(
            settings,
            confirmed=True,
            expected_database_name="super_biz_agent",
            expected_milvus_collection="rag_f0_dev",
        )
        == "expected_database_name_not_f0_scoped"
    )
    assert (
        _static_preflight(
            settings,
            confirmed=True,
            expected_database_name="superbiz_rag_f0_dev",
            expected_milvus_collection="production_chunks",
        )
        == "expected_milvus_collection_not_f0_scoped"
    )


class _ScalarRows:
    def __init__(self, values) -> None:
        self._values = values

    def scalars(self):
        return self

    def all(self):
        return self._values


class _PreflightConnection:
    def __init__(
        self,
        *,
        database="superbiz_rag_f0_dev",
        marker="dataset-sha",
        migrations=None,
        knowledge_bases=0,
        documents=0,
    ) -> None:
        self.database = database
        self.marker = marker
        self.migrations = [ALEMBIC_HEAD] if migrations is None else migrations
        self.knowledge_bases = knowledge_bases
        self.documents = documents

    async def scalar(self, statement, parameters=None):
        del parameters
        sql = str(statement)
        if "current_database" in sql:
            return self.database
        if "rag_f0_evaluation_marker" in sql:
            return self.marker
        if "rag_knowledge_base" in sql:
            return self.knowledge_bases
        if "rag_document" in sql:
            return self.documents
        raise AssertionError(sql)

    async def execute(self, statement):
        assert "alembic_version" in str(statement)
        return _ScalarRows(self.migrations)


class _ConnectionContext:
    def __init__(self, connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _PreflightEngine:
    def __init__(self, connection) -> None:
        self.connection = connection

    def connect(self):
        return _ConnectionContext(self.connection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connection", "failure"),
    [
        (_PreflightConnection(database="wrong"), "actual_database_name_mismatch"),
        (_PreflightConnection(marker="wrong"), "dedicated_database_marker_mismatch"),
        (_PreflightConnection(migrations=["old"]), "alembic_head_mismatch"),
        (_PreflightConnection(knowledge_bases=1), "rag_tables_not_empty"),
        (_PreflightConnection(documents=1), "rag_tables_not_empty"),
    ],
)
async def test_postgres_preflight_fails_closed(connection, failure) -> None:
    with pytest.raises(RagF0SafetyError, match=failure):
        await _validate_postgres_preflight(
            _PreflightEngine(connection),
            expected_database_name="superbiz_rag_f0_dev",
            dataset_sha256="dataset-sha",
        )


@pytest.mark.asyncio
async def test_postgres_preflight_accepts_exact_empty_marked_database() -> None:
    await _validate_postgres_preflight(
        _PreflightEngine(_PreflightConnection()),
        expected_database_name="superbiz_rag_f0_dev",
        dataset_sha256="dataset-sha",
    )


class _MilvusClient:
    def __init__(self, *, exists: bool) -> None:
        self.exists = exists
        self.dropped = []
        self.closed = False

    def has_collection(self, collection_name):
        assert collection_name == "rag_f0_dev"
        return self.exists

    def drop_collection(self, collection_name):
        self.dropped.append(collection_name)
        self.exists = False

    def close(self):
        self.closed = True


def test_milvus_preflight_rejects_existing_collection_and_closes_client() -> None:
    client = _MilvusClient(exists=True)
    with pytest.raises(RagF0SafetyError, match="milvus_collection_must_not_exist"):
        _validate_milvus_preflight(
            uri="test.db",
            token=None,
            collection_name="rag_f0_dev",
            client_factory=lambda **kwargs: client,
        )
    assert client.closed


def test_owned_milvus_collection_cleanup_drops_and_closes() -> None:
    client = _MilvusClient(exists=True)
    _drop_owned_collection(
        uri="test.db",
        token=None,
        collection_name="rag_f0_dev",
        client_factory=lambda **kwargs: client,
    )
    assert client.dropped == ["rag_f0_dev"]
    assert client.closed


@pytest.mark.asyncio
async def test_postgres_cleanup_deletes_documents_before_knowledge_base() -> None:
    statements = []

    class Session:
        async def execute(self, statement):
            statements.append(str(statement))

        async def commit(self):
            statements.append("COMMIT")

        async def scalar(self, statement):
            statements.append(str(statement))
            return 0

    await _cleanup_postgres(lambda: _ConnectionContext(Session()))
    assert statements[0].startswith("DELETE FROM rag_document")
    assert statements[1].startswith("DELETE FROM rag_knowledge_base")
    assert statements[2].startswith("SELECT count(*)")
    assert "rag_document" in statements[2]
    assert statements[3].startswith("SELECT count(*)")
    assert "rag_knowledge_base" in statements[3]
    assert statements[4] == "COMMIT"


@pytest.mark.asyncio
async def test_real_entrypoint_cleanup_failure_withholds_completed_report(
    dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://localhost/superbiz_rag_f0_dev",
        rag_enabled=True,
        rag_fixture_mode=False,
        rag_rerank_enabled=False,
        rag_embedding_api_key="test-only",
        rag_milvus_uri="test.db",
        rag_milvus_collection="rag_f0_dev",
    )
    candidate = await RagF0Runner(
        retrieval_service=_GoldRetrievalService(dataset),
        dataset=dataset,
        bootstrap_samples=100,
    ).run_dev()
    captured = []

    class Resource:
        async def aclose(self):
            return None

    class Engine:
        async def dispose(self):
            return None

    class KnowledgeBases:
        def __init__(self, sessionmaker):
            del sessionmaker

        async def create_default(self, **kwargs):
            del kwargs
            return type("KnowledgeBase", (), {"id": RAG_F0_KNOWLEDGE_BASE_ID})()

    class Documents:
        def __init__(self, sessionmaker):
            del sessionmaker

    class Ingestion:
        def __init__(self, **kwargs):
            del kwargs

        async def ingest_sync(self, request):
            del request

    class Runner:
        def __init__(self, **kwargs):
            del kwargs

        async def run_dev(self):
            return candidate

    async def postgres_preflight(*args, **kwargs):
        del args, kwargs

    async def cleanup_failure(sessionmaker):
        del sessionmaker
        raise RuntimeError

    monkeypatch.setattr(baseline_entrypoint, "Settings", lambda: settings)
    monkeypatch.setattr(baseline_entrypoint, "create_engine", lambda url: Engine())
    monkeypatch.setattr(baseline_entrypoint, "create_sessionmaker", lambda engine: object())
    monkeypatch.setattr(baseline_entrypoint, "_validate_postgres_preflight", postgres_preflight)
    monkeypatch.setattr(baseline_entrypoint, "_validate_milvus_preflight", lambda **kwargs: None)
    monkeypatch.setattr(baseline_entrypoint, "build_rag_embedding_service", lambda settings: Resource())
    monkeypatch.setattr(baseline_entrypoint, "MilvusHybridChunkStore", lambda **kwargs: Resource())
    monkeypatch.setattr(baseline_entrypoint, "RagKnowledgeBaseRepository", KnowledgeBases)
    monkeypatch.setattr(baseline_entrypoint, "RagDocumentRepository", Documents)
    monkeypatch.setattr(baseline_entrypoint, "RagIngestionService", Ingestion)
    monkeypatch.setattr(baseline_entrypoint, "build_real_rag_retrieval_service", lambda settings: Resource())
    monkeypatch.setattr(baseline_entrypoint, "RagF0Runner", Runner)
    monkeypatch.setattr(baseline_entrypoint, "_cleanup_postgres", cleanup_failure)
    monkeypatch.setattr(baseline_entrypoint, "_drop_owned_collection", lambda **kwargs: None)
    monkeypatch.setattr(
        baseline_entrypoint,
        "write_rag_f0_report",
        lambda report, output: captured.append((report, output)),
    )

    output = tmp_path / "report.json"
    exit_code = await baseline_entrypoint.run(
        dataset_path=dataset.root,
        output=output,
        confirm_dedicated_postgres=True,
        expected_database_name="superbiz_rag_f0_dev",
        expected_milvus_collection="rag_f0_dev",
    )

    assert exit_code == 2
    report, reported_output = captured[-1]
    assert reported_output == output
    assert report.status == "infrastructure_pending"
    assert report.metrics == {}
    assert report.slice_metrics == {}
    assert report.ablations["hybrid"] == "infrastructure_pending: postgres_cleanup_failed"


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
