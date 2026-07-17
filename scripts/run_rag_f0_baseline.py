from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from superbiz_agent.config import Settings
from superbiz_agent.evals.rag_cases import (
    RAG_F0_TENANT_ID,
    default_rag_f0_dataset_path,
    load_rag_f0_dataset,
)
from superbiz_agent.evals.rag_runner import (
    RagF0Runner,
    infrastructure_pending_report,
    write_rag_f0_report,
)
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.repositories.rag import (
    RagDocumentRepository,
    RagKnowledgeBaseRepository,
)
from superbiz_agent.rag.chunking import RagDocumentChunker
from superbiz_agent.rag.embedding import build_rag_embedding_service
from superbiz_agent.rag.ingestion import RagIngestionService
from superbiz_agent.rag.milvus_store import MilvusHybridChunkStore
from superbiz_agent.rag.models import RagDocumentContentType, RagIngestionRequest
from superbiz_agent.rag.runtime import build_real_rag_retrieval_service


async def run(*, dataset_path: Path, output: Path, confirm_dedicated_postgres: bool) -> int:
    dataset = load_rag_f0_dataset(dataset_path)
    settings = Settings()
    failure = _preflight(settings, confirm_dedicated_postgres)
    if failure is not None:
        write_rag_f0_report(infrastructure_pending_report(dataset, failure), output)
        return 2

    eval_settings = settings.model_copy(
        update={
            "rag_enabled": True,
            "rag_fixture_mode": False,
            "rag_hybrid_top_k": 10,
            "rag_final_top_k": 10,
            "rag_rerank_enabled": False,
        }
    )
    engine = create_engine(eval_settings.database_url)
    sessionmaker = create_sessionmaker(engine)
    embedding = build_rag_embedding_service(eval_settings)
    store = MilvusHybridChunkStore(
        uri=eval_settings.rag_milvus_uri,
        token=eval_settings.rag_milvus_token,
        collection_name=eval_settings.rag_milvus_collection,
        dimension=eval_settings.rag_embedding_dimension,
    )
    try:
        knowledge_bases = RagKnowledgeBaseRepository(sessionmaker)
        documents = RagDocumentRepository(sessionmaker)
        knowledge_base = await knowledge_bases.create_default(
            tenant_id=RAG_F0_TENANT_ID,
            name="RAG F0 Evaluation Knowledge Base",
            created_by="rag-f0-runner",
            description="Pinned F0 dev retrieval corpus; all documents are active.",
        )
        ingestion = RagIngestionService(
            settings=eval_settings,
            knowledge_base_repository=knowledge_bases,
            document_repository=documents,
            chunker=RagDocumentChunker(),
            embedding_service=embedding,
            chunk_store=store,
        )
        for document in dataset.corpus:
            await ingestion.ingest_sync(
                RagIngestionRequest(
                    tenant_id=RAG_F0_TENANT_ID,
                    knowledge_base_id=knowledge_base.id,
                    document_name=document.document_name,
                    source_uri=f"rag-f0://{document.document_id}",
                    content=document.text,
                    content_type=RagDocumentContentType.MARKDOWN,
                    created_by="rag-f0-runner",
                    parser="rag-f0-pinned-markdown",
                    parser_version=dataset.manifest.dataset_version,
                )
            )
    except Exception as exc:
        write_rag_f0_report(infrastructure_pending_report(dataset, exc.__class__.__name__), output)
        return 2
    finally:
        await store.aclose()
        await embedding.aclose()
        await engine.dispose()

    runtime = build_real_rag_retrieval_service(eval_settings)
    try:
        report = await RagF0Runner(
            retrieval_service=runtime,
            dataset=dataset,
        ).run_dev()
        write_rag_f0_report(report, output)
        return 0 if report.status == "completed" else 2
    finally:
        await runtime.aclose()


def _preflight(settings: Settings, confirmed: bool) -> str | None:
    if not confirmed:
        return "dedicated_postgres_not_confirmed"
    if not settings.database_url.startswith("postgresql+"):
        return "dedicated_postgres_not_configured"
    if not settings.rag_enabled:
        return "rag_enabled_false"
    if settings.rag_fixture_mode:
        return "rag_fixture_mode_true"
    if settings.rag_rerank_enabled:
        return "rerank_must_be_disabled"
    if not (settings.rag_embedding_api_key or settings.model_api_key):
        return "real_embedding_credentials_unavailable"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=default_rag_f0_dataset_path())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/rag_f0/dev-baseline.json"),
    )
    parser.add_argument("--confirm-dedicated-postgres", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run(
                dataset_path=args.dataset,
                output=args.output,
                confirm_dedicated_postgres=args.confirm_dedicated_postgres,
            )
        )
    )


if __name__ == "__main__":
    main()
