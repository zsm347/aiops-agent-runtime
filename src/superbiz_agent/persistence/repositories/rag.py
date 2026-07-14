from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import re
from typing import Any
from uuid import uuid4

from sqlalchemy import Select, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from superbiz_agent.config import RAG_TOP_K_MAX
from superbiz_agent.persistence.models import RagDocument, RagKnowledgeBase


class RagPersistenceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RagKnowledgeBaseNotActiveError(RagPersistenceError):
    def __init__(self) -> None:
        super().__init__("knowledge_base_not_active", "knowledge base is not active for tenant")


class RagKnowledgeBaseContractError(RagPersistenceError):
    def __init__(self) -> None:
        super().__init__(
            "knowledge_base_contract_error",
            "multiple active default knowledge bases violate the repository contract",
        )


class RagDocumentArchivedError(RagPersistenceError):
    def __init__(self) -> None:
        super().__init__("document_archived", "archived document cannot be claimed")


class RagClaimLostError(RagPersistenceError):
    def __init__(self) -> None:
        super().__init__("claim_lost", "document ingestion claim was lost")


class RagDocumentClaimStatus(str, Enum):
    CLAIMED = "claimed"
    DUPLICATE_SKIPPED = "duplicate_skipped"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True, slots=True)
class RagKnowledgeBaseRecord:
    id: str
    tenant_id: str
    name: str
    description: str | None
    status: str
    is_default: bool
    created_by: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RagDocumentClaim:
    status: RagDocumentClaimStatus
    document_id: str
    content_hash: str
    chunk_count: int
    claim_token: str | None = None


class RagKnowledgeBaseRepository:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self.sessionmaker = sessionmaker

    async def create_default(
        self,
        *,
        tenant_id: str,
        name: str,
        created_by: str,
        description: str | None = None,
    ) -> RagKnowledgeBaseRecord:
        _require_values(tenant_id=tenant_id, name=name, created_by=created_by)
        statement = (
            insert(RagKnowledgeBase)
            .values(
                id=str(uuid4()),
                tenant_id=tenant_id,
                name=name,
                description=description,
                status="active",
                is_default=True,
                created_by=created_by,
            )
            .on_conflict_do_nothing(
                index_elements=[RagKnowledgeBase.tenant_id],
                index_where=text("status = 'active' AND is_default = TRUE"),
            )
            .returning(RagKnowledgeBase)
        )
        try:
            async with self.sessionmaker() as session:
                row = (await session.execute(statement)).scalar_one_or_none()
                if row is None:
                    row = await session.scalar(
                        self.default_statement(tenant_id=tenant_id).with_for_update()
                    )
                if row is None:
                    raise RagPersistenceError(
                        "default_knowledge_base_conflict",
                        "default knowledge base conflict could not be resolved",
                    )
                record = _knowledge_base_record(row)
                await session.commit()
                return record
        except RagPersistenceError:
            raise
        except SQLAlchemyError as exc:
            raise RagPersistenceError(
                "knowledge_base_store_error", "failed to create default knowledge base"
            ) from exc

    async def get_active_for_tenant(
        self, *, tenant_id: str, knowledge_base_id: str
    ) -> RagKnowledgeBaseRecord | None:
        _require_values(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
        try:
            async with self.sessionmaker() as session:
                row = await session.scalar(
                    self.active_statement(
                        tenant_id=tenant_id,
                        knowledge_base_id=knowledge_base_id,
                    )
                )
                return _knowledge_base_record(row) if row is not None else None
        except SQLAlchemyError as exc:
            raise RagPersistenceError(
                "knowledge_base_store_error", "failed to read active knowledge base"
            ) from exc

    async def get_default_for_tenant(self, *, tenant_id: str) -> RagKnowledgeBaseRecord | None:
        _require_values(tenant_id=tenant_id)
        try:
            async with self.sessionmaker() as session:
                result = await session.execute(self.default_statement(tenant_id=tenant_id))
                rows = result.scalars().all()
                if len(rows) > 1:
                    raise RagKnowledgeBaseContractError()
                return _knowledge_base_record(rows[0]) if rows else None
        except RagPersistenceError:
            raise
        except SQLAlchemyError as exc:
            raise RagPersistenceError(
                "knowledge_base_store_error", "failed to read default knowledge base"
            ) from exc

    @staticmethod
    def active_statement(
        *, tenant_id: str, knowledge_base_id: str
    ) -> Select[tuple[RagKnowledgeBase]]:
        _require_values(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
        return select(RagKnowledgeBase).where(
            RagKnowledgeBase.tenant_id == tenant_id,
            RagKnowledgeBase.id == knowledge_base_id,
            RagKnowledgeBase.status == "active",
        )

    @staticmethod
    def default_statement(*, tenant_id: str) -> Select[tuple[RagKnowledgeBase]]:
        _require_values(tenant_id=tenant_id)
        return select(RagKnowledgeBase).where(
            RagKnowledgeBase.tenant_id == tenant_id,
            RagKnowledgeBase.status == "active",
            RagKnowledgeBase.is_default.is_(True),
        ).limit(2)


class RagDocumentRepository:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self.sessionmaker = sessionmaker

    async def get_document_statuses(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        document_ids: Sequence[str],
    ) -> dict[str, str]:
        _require_values(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
        values = tuple(dict.fromkeys(document_ids))
        if not values:
            return {}
        if len(values) > RAG_TOP_K_MAX:
            raise ValueError("document_ids exceeds the repository safety limit")
        _require_values(**{f"document_ids[{index}]": value for index, value in enumerate(values)})
        try:
            async with self.sessionmaker() as session:
                rows = (await session.execute(
                    self.statuses_statement(
                        tenant_id=tenant_id,
                        knowledge_base_id=knowledge_base_id,
                        document_ids=values,
                    )
                )).all()
                return {str(document_id): str(status) for document_id, status in rows}
        except SQLAlchemyError as exc:
            raise RagPersistenceError(
                "document_store_error", "failed to read document statuses"
            ) from exc

    async def claim_for_ingestion(
        self,
        *,
        tenant_id: str,
        knowledge_base_id: str,
        source_uri: str,
        document_name: str,
        content_type: str,
        parser: str,
        parser_version: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_version: str,
        embedding_dimension: int,
        content_hash: str,
        created_by: str,
        stale_after_seconds: int,
    ) -> RagDocumentClaim:
        _require_values(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            source_uri=source_uri,
            document_name=document_name,
            content_type=content_type,
            parser=parser,
            parser_version=parser_version,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            content_hash=content_hash,
            created_by=created_by,
        )
        if embedding_dimension < 1 or stale_after_seconds < 1:
            raise ValueError("embedding_dimension and stale_after_seconds must be positive")
        _require_sha256_content_hash(content_hash)

        document_id = str(uuid4())
        claim_token = str(uuid4())
        try:
            async with self.sessionmaker() as session:
                knowledge_base = await session.scalar(
                    RagKnowledgeBaseRepository.active_statement(
                        tenant_id=tenant_id,
                        knowledge_base_id=knowledge_base_id,
                    ).with_for_update()
                )
                if knowledge_base is None:
                    raise RagKnowledgeBaseNotActiveError()

                statement = (
                    insert(RagDocument)
                    .values(
                        id=document_id,
                        tenant_id=tenant_id,
                        knowledge_base_id=knowledge_base_id,
                        source_uri=source_uri,
                        document_name=document_name,
                        content_type=content_type,
                        parser=parser,
                        parser_version=parser_version,
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                        embedding_version=embedding_version,
                        embedding_dimension=embedding_dimension,
                        status="indexing",
                        content_hash=content_hash,
                        chunk_count=0,
                        claim_token=claim_token,
                        claimed_at=func.current_timestamp(),
                        created_by=created_by,
                        updated_at=func.current_timestamp(),
                    )
                    .on_conflict_do_nothing(
                        index_elements=[
                            RagDocument.tenant_id,
                            RagDocument.knowledge_base_id,
                            RagDocument.content_hash,
                        ]
                    )
                    .returning(RagDocument)
                )
                row = (await session.execute(statement)).scalar_one_or_none()
                if row is None:
                    row = await session.scalar(
                        self.existing_statement(
                            tenant_id=tenant_id,
                            knowledge_base_id=knowledge_base_id,
                            content_hash=content_hash,
                        ).with_for_update()
                    )
                    if row is None:
                        raise RagPersistenceError(
                            "document_claim_conflict",
                            "document claim conflict could not be resolved",
                        )
                    claim = await self._claim_existing(
                        session=session,
                        row=row,
                        stale_after_seconds=stale_after_seconds,
                    )
                else:
                    claim = _document_claim(row, RagDocumentClaimStatus.CLAIMED)

                await session.commit()
                return claim
        except RagPersistenceError:
            raise
        except SQLAlchemyError as exc:
            raise RagPersistenceError("document_store_error", "failed to claim document") from exc

    async def _claim_existing(
        self,
        *,
        session: AsyncSession,
        row: RagDocument,
        stale_after_seconds: int,
    ) -> RagDocumentClaim:
        if row.status == "active":
            return _document_claim(row, RagDocumentClaimStatus.DUPLICATE_SKIPPED)
        if row.status == "archived":
            raise RagDocumentArchivedError()
        if row.status == "indexing":
            database_now = await session.scalar(select(func.current_timestamp()))
            if database_now is None or row.claimed_at is None:
                raise RagPersistenceError("invalid_claim_state", "document claim state is invalid")
            if row.claimed_at > database_now - timedelta(seconds=stale_after_seconds):
                return _document_claim(row, RagDocumentClaimStatus.IN_PROGRESS)
        elif row.status != "failed":
            raise RagPersistenceError("invalid_document_status", "document status is invalid")

        row.status = "indexing"
        row.claim_token = str(uuid4())
        row.claimed_at = func.current_timestamp()
        row.updated_at = func.current_timestamp()
        row.chunk_count = 0
        row.failure_code = None
        row.failure_message = None
        return _document_claim(row, RagDocumentClaimStatus.CLAIMED)

    async def refresh_claim(self, *, tenant_id: str, document_id: str, claim_token: str) -> None:
        await self._execute_fenced_update(
            self.refresh_claim_statement(
                tenant_id=tenant_id,
                document_id=document_id,
                claim_token=claim_token,
            )
        )

    async def mark_active(
        self,
        *,
        tenant_id: str,
        document_id: str,
        claim_token: str,
        chunk_count: int,
    ) -> None:
        if chunk_count < 0:
            raise ValueError("chunk_count must be non-negative")
        await self._execute_fenced_update(
            self.mark_active_statement(
                tenant_id=tenant_id,
                document_id=document_id,
                claim_token=claim_token,
                chunk_count=chunk_count,
            )
        )

    async def mark_failed(
        self,
        *,
        tenant_id: str,
        document_id: str,
        claim_token: str,
        failure_code: str,
        sanitized_message: str,
    ) -> None:
        _require_values(failure_code=failure_code, sanitized_message=sanitized_message)
        await self._execute_fenced_update(
            self.mark_failed_statement(
                tenant_id=tenant_id,
                document_id=document_id,
                claim_token=claim_token,
                failure_code=failure_code[:100],
                sanitized_message=sanitized_message[:1000],
            )
        )

    async def _execute_fenced_update(self, statement: Any) -> None:
        try:
            async with self.sessionmaker() as session:
                result = await session.execute(statement)
                if result.rowcount != 1:
                    raise RagClaimLostError()
                await session.commit()
        except RagPersistenceError:
            raise
        except SQLAlchemyError as exc:
            raise RagPersistenceError("document_store_error", "failed to update document") from exc

    @staticmethod
    def existing_statement(
        *, tenant_id: str, knowledge_base_id: str, content_hash: str
    ) -> Select[tuple[RagDocument]]:
        _require_values(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            content_hash=content_hash,
        )
        return select(RagDocument).where(
            RagDocument.tenant_id == tenant_id,
            RagDocument.knowledge_base_id == knowledge_base_id,
            RagDocument.content_hash == content_hash,
        )

    @staticmethod
    def statuses_statement(
        *, tenant_id: str, knowledge_base_id: str, document_ids: Sequence[str]
    ) -> Select[tuple[str, str]]:
        _require_values(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
        values = tuple(dict.fromkeys(document_ids))
        if not values:
            raise ValueError("document_ids must not be empty")
        if len(values) > RAG_TOP_K_MAX:
            raise ValueError("document_ids exceeds the repository safety limit")
        _require_values(**{f"document_ids[{index}]": value for index, value in enumerate(values)})
        return select(RagDocument.id, RagDocument.status).where(
            RagDocument.tenant_id == tenant_id,
            RagDocument.knowledge_base_id == knowledge_base_id,
            RagDocument.id.in_(values),
        )

    @staticmethod
    def refresh_claim_statement(*, tenant_id: str, document_id: str, claim_token: str):
        return RagDocumentRepository._fenced_update(
            tenant_id=tenant_id,
            document_id=document_id,
            claim_token=claim_token,
        ).values(
            claimed_at=func.current_timestamp(),
            updated_at=func.current_timestamp(),
        )

    @staticmethod
    def mark_active_statement(
        *, tenant_id: str, document_id: str, claim_token: str, chunk_count: int
    ):
        return RagDocumentRepository._fenced_update(
            tenant_id=tenant_id,
            document_id=document_id,
            claim_token=claim_token,
        ).values(
            status="active",
            chunk_count=chunk_count,
            claim_token=None,
            claimed_at=None,
            failure_code=None,
            failure_message=None,
            updated_at=func.current_timestamp(),
        )

    @staticmethod
    def mark_failed_statement(
        *,
        tenant_id: str,
        document_id: str,
        claim_token: str,
        failure_code: str,
        sanitized_message: str,
    ):
        return RagDocumentRepository._fenced_update(
            tenant_id=tenant_id,
            document_id=document_id,
            claim_token=claim_token,
        ).values(
            status="failed",
            chunk_count=0,
            claim_token=None,
            claimed_at=None,
            failure_code=failure_code,
            failure_message=sanitized_message,
            updated_at=func.current_timestamp(),
        )

    @staticmethod
    def _fenced_update(*, tenant_id: str, document_id: str, claim_token: str):
        _require_values(
            tenant_id=tenant_id,
            document_id=document_id,
            claim_token=claim_token,
        )
        return update(RagDocument).where(
            RagDocument.tenant_id == tenant_id,
            RagDocument.id == document_id,
            RagDocument.status == "indexing",
            RagDocument.claim_token == claim_token,
        )


def _knowledge_base_record(row: RagKnowledgeBase) -> RagKnowledgeBaseRecord:
    return RagKnowledgeBaseRecord(
        id=row.id,
        tenant_id=row.tenant_id,
        name=row.name,
        description=row.description,
        status=row.status,
        is_default=row.is_default,
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _document_claim(row: RagDocument, status: RagDocumentClaimStatus) -> RagDocumentClaim:
    return RagDocumentClaim(
        status=status,
        document_id=row.id,
        content_hash=row.content_hash,
        chunk_count=row.chunk_count,
        claim_token=row.claim_token if status is RagDocumentClaimStatus.CLAIMED else None,
    )


def _require_values(**values: object) -> None:
    missing = [
        name for name, value in values.items() if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        raise ValueError(f"required values are missing: {', '.join(missing)}")


def _require_sha256_content_hash(content_hash: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
        raise ValueError("content_hash must be a 64-character lowercase hexadecimal SHA-256")
