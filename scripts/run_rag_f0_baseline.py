from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient
from sqlalchemy import delete, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from superbiz_agent.config import Settings
from superbiz_agent.evals.rag_cases import (
    RAG_F0_TENANT_ID,
    default_rag_f0_dataset_path,
    load_rag_f0_dataset,
)
from superbiz_agent.evals.rag_runner import (
    RagF0Report,
    RagF0Runner,
    infrastructure_pending_report,
    write_rag_f0_report,
)
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.models import RagDocument, RagKnowledgeBase
from superbiz_agent.persistence.repositories.rag import (
    RagDocumentRepository,
    RagKnowledgeBaseRepository,
)
from superbiz_agent.rag.chunking import RagDocumentChunker
from superbiz_agent.rag.embedding import build_rag_embedding_service
from superbiz_agent.rag.ingestion import RagIngestionError, RagIngestionService
from superbiz_agent.rag.milvus_store import MilvusHybridChunkStore
from superbiz_agent.rag.models import RagDocumentContentType, RagIngestionRequest
from superbiz_agent.rag.runtime import build_real_rag_retrieval_service


DATABASE_NAME_PREFIX = "superbiz_rag_f0_"
MILVUS_COLLECTION_PREFIX = "rag_f0_"
DATABASE_MARKER_TABLE = "rag_f0_evaluation_marker"
DATABASE_MARKER_KEY = "rag-f0-dedicated-database"
ALEMBIC_HEAD = "20260714_01"
POSTGRESQL_VERSION = "16.14"
PGVECTOR_VERSION = "0.8.5"
EMBEDDING_BATCH_SIZE = 10


class RagF0SafetyError(RuntimeError):
    """Raised when the real baseline cannot prove ownership of its infrastructure."""


async def run(
    *,
    dataset_path: Path,
    output: Path,
    confirm_dedicated_postgres: bool,
    expected_database_name: str | None,
    expected_milvus_collection: str | None,
    allow_model_gateway_embedding_credentials: bool = False,
) -> int:
    dataset = load_rag_f0_dataset(dataset_path)
    settings = Settings()
    failure = _static_preflight(
        settings,
        confirmed=confirm_dedicated_postgres,
        expected_database_name=expected_database_name,
        expected_milvus_collection=expected_milvus_collection,
        allow_model_gateway_embedding_credentials=allow_model_gateway_embedding_credentials,
    )
    if failure is not None:
        write_rag_f0_report(infrastructure_pending_report(dataset, failure), output)
        return 2

    assert expected_database_name is not None
    assert expected_milvus_collection is not None
    use_model_gateway_credentials = allow_model_gateway_embedding_credentials and not (
        settings.rag_embedding_api_key and settings.rag_embedding_base_url
    )
    setting_overrides: dict[str, Any] = {
            "rag_enabled": True,
            "rag_fixture_mode": False,
            "rag_hybrid_top_k": 10,
            "rag_final_top_k": 10,
            "rag_rerank_enabled": False,
            "rag_embedding_batch_size": EMBEDDING_BATCH_SIZE,
    }
    if use_model_gateway_credentials:
        setting_overrides.update(
            rag_embedding_api_key=settings.model_api_key,
            rag_embedding_base_url=settings.model_base_url,
        )
    eval_settings = settings.model_copy(update=setting_overrides)
    engine = create_engine(eval_settings.database_url)
    sessionmaker = create_sessionmaker(engine)
    embedding: Any | None = None
    store: MilvusHybridChunkStore | None = None
    runtime: Any | None = None
    candidate_report: RagF0Report | None = None
    failure_codes: list[str] = []
    cleanup_authorized = False
    infrastructure_versions: dict[str, str] = {}
    embedding_identity: dict[str, str | int] = {}

    try:
        infrastructure_versions = await _validate_postgres_preflight(
            engine,
            expected_database_name=expected_database_name,
            dataset_sha256=dataset.manifest.dataset_sha256,
        )
        await asyncio.to_thread(
            _validate_milvus_preflight,
            uri=eval_settings.rag_milvus_uri,
            token=eval_settings.rag_milvus_token,
            collection_name=expected_milvus_collection,
        )
        # The collection was proven absent. From this point, this invocation owns any
        # collection with the exact expected name and may remove it in finally.
        cleanup_authorized = True

        embedding = build_rag_embedding_service(eval_settings)
        embedding_identity = {
            "provider": embedding.provider,
            "model": embedding.model,
            "version": embedding.version,
            "dimension": embedding.dimension,
            "batch_size": embedding.batch_size,
            "credential_source": (
                "model_gateway_authorized"
                if use_model_gateway_credentials
                else "rag_specific"
            ),
        }
        store = MilvusHybridChunkStore(
            uri=eval_settings.rag_milvus_uri,
            token=eval_settings.rag_milvus_token,
            collection_name=expected_milvus_collection,
            dimension=eval_settings.rag_embedding_dimension,
        )
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

        if await _close_resource(store.aclose, failure_codes, "ingestion_store_close_failed"):
            store = None
        if await _close_resource(embedding.aclose, failure_codes, "embedding_close_failed"):
            embedding = None
        if failure_codes:
            raise RagF0SafetyError("resource_close_failed_before_retrieval")

        runtime = build_real_rag_retrieval_service(eval_settings)
        candidate_report = await RagF0Runner(
            retrieval_service=runtime,
            dataset=dataset,
            embedding_identity=embedding_identity,
            infrastructure_versions=infrastructure_versions,
        ).run_dev()
    except Exception as exc:
        failure_codes.append(_failure_code(exc))
    finally:
        if runtime is not None:
            await _close_resource(runtime.aclose, failure_codes, "runtime_close_failed")
        if store is not None:
            await _close_resource(store.aclose, failure_codes, "ingestion_store_close_failed")
        if embedding is not None:
            await _close_resource(embedding.aclose, failure_codes, "embedding_close_failed")
        if cleanup_authorized:
            try:
                await _cleanup_postgres(sessionmaker)
            except Exception:
                failure_codes.append("postgres_cleanup_failed")
            try:
                await asyncio.to_thread(
                    _drop_owned_collection,
                    uri=eval_settings.rag_milvus_uri,
                    token=eval_settings.rag_milvus_token,
                    collection_name=expected_milvus_collection,
                )
            except Exception:
                failure_codes.append("milvus_cleanup_failed")
        await _close_resource(engine.dispose, failure_codes, "engine_dispose_failed")

    if candidate_report is not None and not failure_codes:
        write_rag_f0_report(candidate_report, output)
        return 0 if candidate_report.status == "completed" else 2

    failure = ",".join(dict.fromkeys(failure_codes)) or "baseline_not_completed"
    write_rag_f0_report(infrastructure_pending_report(dataset, failure), output)
    return 2


def _static_preflight(
    settings: Settings,
    *,
    confirmed: bool,
    expected_database_name: str | None,
    expected_milvus_collection: str | None,
    allow_model_gateway_embedding_credentials: bool = False,
) -> str | None:
    if not confirmed:
        return "dedicated_postgres_not_confirmed"
    if not expected_database_name:
        return "expected_database_name_required"
    if not expected_database_name.startswith(DATABASE_NAME_PREFIX):
        return "expected_database_name_not_f0_scoped"
    if not expected_milvus_collection:
        return "expected_milvus_collection_required"
    if not expected_milvus_collection.startswith(MILVUS_COLLECTION_PREFIX):
        return "expected_milvus_collection_not_f0_scoped"
    try:
        configured_database = make_url(settings.database_url).database
    except Exception:
        return "dedicated_postgres_not_configured"
    if not settings.database_url.startswith("postgresql+") or configured_database is None:
        return "dedicated_postgres_not_configured"
    if configured_database != expected_database_name:
        return "configured_database_name_mismatch"
    if settings.rag_milvus_collection != expected_milvus_collection:
        return "configured_milvus_collection_mismatch"
    if not settings.rag_enabled:
        return "rag_enabled_false"
    if settings.rag_fixture_mode:
        return "rag_fixture_mode_true"
    if settings.rag_rerank_enabled:
        return "rerank_must_be_disabled"
    rag_credentials_available = bool(
        settings.rag_embedding_api_key and settings.rag_embedding_base_url
    )
    shared_credentials_available = bool(settings.model_api_key and settings.model_base_url)
    if not rag_credentials_available and not (
        allow_model_gateway_embedding_credentials and shared_credentials_available
    ):
        return "rag_embedding_credentials_unavailable"
    provider = settings.rag_embedding_provider.casefold()
    if any(marker in provider for marker in ("deterministic", "fixture", "stub")):
        return "real_rag_embedding_provider_required"
    return None


async def _validate_postgres_preflight(
    engine: AsyncEngine,
    *,
    expected_database_name: str,
    dataset_sha256: str,
) -> dict[str, str]:
    try:
        async with engine.connect() as connection:
            actual_database = await connection.scalar(text("SELECT current_database()"))
            if actual_database != expected_database_name:
                raise RagF0SafetyError("actual_database_name_mismatch")

            postgresql_version_number = await connection.scalar(text("SHOW server_version_num"))
            if postgresql_version_number != "160014":
                raise RagF0SafetyError("postgresql_version_mismatch")

            pgvector_version = await connection.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            if pgvector_version != PGVECTOR_VERSION:
                raise RagF0SafetyError("pgvector_version_mismatch")

            marker = await connection.scalar(
                text(
                    f"SELECT dataset_sha256 FROM {DATABASE_MARKER_TABLE} "
                    "WHERE marker_key = :marker_key"
                ),
                {"marker_key": DATABASE_MARKER_KEY},
            )
            if marker != dataset_sha256:
                raise RagF0SafetyError("dedicated_database_marker_mismatch")

            migration_rows = (
                await connection.execute(text("SELECT version_num FROM alembic_version"))
            ).scalars().all()
            if migration_rows != [ALEMBIC_HEAD]:
                raise RagF0SafetyError("alembic_head_mismatch")

            knowledge_base_count = await connection.scalar(
                select(func.count()).select_from(RagKnowledgeBase)
            )
            document_count = await connection.scalar(select(func.count()).select_from(RagDocument))
            if knowledge_base_count != 0 or document_count != 0:
                raise RagF0SafetyError("rag_tables_not_empty")
            return {
                "postgresql": POSTGRESQL_VERSION,
                "pgvector": str(pgvector_version),
                "milvus_lite": version("milvus-lite"),
            }
    except RagF0SafetyError:
        raise
    except Exception:
        raise RagF0SafetyError("postgres_preflight_failed") from None


def _validate_milvus_preflight(
    *,
    uri: str,
    token: str | None,
    collection_name: str,
    client_factory: Callable[..., Any] = MilvusClient,
) -> None:
    client = client_factory(uri=uri, token=token or "")
    try:
        if client.has_collection(collection_name):
            raise RagF0SafetyError("milvus_collection_must_not_exist")
    except RagF0SafetyError:
        raise
    except Exception:
        raise RagF0SafetyError("milvus_preflight_failed") from None
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()


async def _cleanup_postgres(sessionmaker: async_sessionmaker[Any]) -> None:
    async with sessionmaker() as session:
        await session.execute(delete(RagDocument).where(RagDocument.tenant_id == RAG_F0_TENANT_ID))
        await session.execute(
            delete(RagKnowledgeBase).where(RagKnowledgeBase.tenant_id == RAG_F0_TENANT_ID)
        )
        remaining_documents = await session.scalar(
            select(func.count())
            .select_from(RagDocument)
            .where(RagDocument.tenant_id == RAG_F0_TENANT_ID)
        )
        remaining_knowledge_bases = await session.scalar(
            select(func.count())
            .select_from(RagKnowledgeBase)
            .where(RagKnowledgeBase.tenant_id == RAG_F0_TENANT_ID)
        )
        if remaining_documents != 0 or remaining_knowledge_bases != 0:
            raise RagF0SafetyError("postgres_cleanup_incomplete")
        await session.commit()


def _drop_owned_collection(
    *,
    uri: str,
    token: str | None,
    collection_name: str,
    client_factory: Callable[..., Any] = MilvusClient,
) -> None:
    client = client_factory(uri=uri, token=token or "")
    try:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
        if client.has_collection(collection_name):
            raise RagF0SafetyError("milvus_cleanup_incomplete")
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()


async def _close_resource(
    close: Callable[[], Awaitable[None]],
    failures: list[str],
    failure_code: str,
) -> bool:
    try:
        await close()
    except Exception:
        failures.append(failure_code)
        return False
    return True


def _failure_code(exc: Exception) -> str:
    if isinstance(exc, RagF0SafetyError):
        return str(exc)
    if isinstance(exc, RagIngestionError):
        return f"RagIngestionError:{exc.code}"
    return exc.__class__.__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=default_rag_f0_dataset_path())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evals/rag_f0/dev-baseline.json"),
    )
    parser.add_argument("--confirm-dedicated-postgres", action="store_true")
    parser.add_argument("--expected-database-name")
    parser.add_argument("--expected-milvus-collection")
    parser.add_argument("--allow-model-gateway-embedding-credentials", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run(
                dataset_path=args.dataset,
                output=args.output,
                confirm_dedicated_postgres=args.confirm_dedicated_postgres,
                expected_database_name=args.expected_database_name,
                expected_milvus_collection=args.expected_milvus_collection,
                allow_model_gateway_embedding_credentials=(
                    args.allow_model_gateway_embedding_credentials
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
