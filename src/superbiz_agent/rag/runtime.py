from __future__ import annotations

import asyncio
import inspect
from typing import Any

from superbiz_agent.config import Settings
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.repositories.rag import (
    RagDocumentRepository,
    RagKnowledgeBaseRepository,
)
from superbiz_agent.rag.embedding import build_rag_embedding_service
from superbiz_agent.rag.milvus_store import MilvusHybridChunkStore
from superbiz_agent.rag.models import RagRetrievalMode, RagRetrievalRequest, RagRetrievalResult
from superbiz_agent.rag.retrieval import (
    MilvusRagRetrievalService,
    RepositoryDefaultKnowledgeBaseResolver,
)


class RealRagRuntime:
    def __init__(
        self,
        *,
        retrieval_service: MilvusRagRetrievalService,
        chunk_store: MilvusHybridChunkStore,
        embedding_service: Any,
        engine: Any,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._chunk_store = chunk_store
        self._embedding_service = embedding_service
        self._engine = engine
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False

    async def search(self, request: RagRetrievalRequest) -> RagRetrievalResult:
        if self._closing or self._closed:
            raise RuntimeError("RAG runtime is closed.")
        return await self._retrieval_service.search(request)

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            task = self._close_task
            if task is None or task.done():
                task = asyncio.create_task(self._close_resources())
                self._close_task = task
        await asyncio.shield(task)

    async def _close_resources(self) -> None:
        failures: list[str] = []
        resources = (
            ("chunk_store", getattr(self._chunk_store, "aclose", None)),
            ("embedding_service", getattr(self._embedding_service, "aclose", None)),
            ("engine", getattr(self._engine, "dispose", None)),
        )
        for label, close in resources:
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except (Exception, asyncio.CancelledError):
                failures.append(label)
        if failures:
            labels = ", ".join(failures)
            raise RuntimeError(f"Failed to close RAG runtime resources: {labels}.") from None
        async with self._close_lock:
            self._closed = True


def build_real_rag_retrieval_service(
    settings: Settings,
    *,
    retrieval_mode: RagRetrievalMode = RagRetrievalMode.HYBRID,
) -> RealRagRuntime:
    embedding = build_rag_embedding_service(settings)
    chunk_store = MilvusHybridChunkStore(
        uri=settings.rag_milvus_uri,
        token=settings.rag_milvus_token,
        collection_name=settings.rag_milvus_collection,
        dimension=settings.rag_embedding_dimension,
    )
    engine = create_engine(settings.database_url)
    sessionmaker = create_sessionmaker(engine)
    knowledge_bases = RagKnowledgeBaseRepository(sessionmaker)
    documents = RagDocumentRepository(sessionmaker)
    resolver = RepositoryDefaultKnowledgeBaseResolver(knowledge_bases)
    retrieval = MilvusRagRetrievalService(
        settings=settings,
        knowledge_base_resolver=resolver,
        document_repository=documents,
        embedding_service=embedding,
        chunk_store=chunk_store,
        retrieval_mode=retrieval_mode,
    )
    return RealRagRuntime(
        retrieval_service=retrieval,
        chunk_store=chunk_store,
        embedding_service=embedding,
        engine=engine,
    )
