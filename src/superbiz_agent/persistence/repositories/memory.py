from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
import math
import re
from typing import Any
from uuid import uuid4

from sqlalchemy import case, func, literal_column, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from superbiz_agent.memory.dedup import canonical_content_hash, stable_tag_union
from superbiz_agent.memory.errors import (
    CoreMemoryContractError,
    MemoryExactConflictUnresolvedError,
    MemoryPersistenceError,
    MemoryStoreContractError,
    MemoryStoreUnavailableError,
)
from superbiz_agent.memory.persistence_contract import (
    ACTIVE_EXACT_INDEX,
    ACTIVE_EXACT_PREDICATE_SQL,
    CORE_UNIQUE_CONSTRAINT,
    M_P2_REQUIRED_CHECKS,
)
from superbiz_agent.memory.ports import CoreContentWriteResult
from superbiz_agent.memory.schemas import (
    CORE_BLOCK_SPECS,
    DEFAULT_CORE_BLOCK_KEYS,
    CoreMemoryBlock,
    LongTermMemory as MemoryRecord,
)
from superbiz_agent.memory.store import ExactMemoryWriteResult, content_hash
from superbiz_agent.persistence.models import AgentCoreMemoryBlock, LongTermMemory


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MEMORY_COLUMNS = (
    LongTermMemory.id,
    LongTermMemory.tenant_id,
    LongTermMemory.user_id,
    LongTermMemory.agent_id,
    LongTermMemory.session_id,
    LongTermMemory.type,
    LongTermMemory.topic,
    LongTermMemory.content,
    LongTermMemory.source,
    LongTermMemory.embedding_model,
    LongTermMemory.embedding_dimension,
    LongTermMemory.embedding_metric,
    LongTermMemory.embedding_version,
    LongTermMemory.created_at,
    LongTermMemory.updated_at,
    LongTermMemory.usage_count,
    LongTermMemory.last_used_at,
    LongTermMemory.status,
    LongTermMemory.archived_at,
    LongTermMemory.archive_reason,
    LongTermMemory.tags,
    LongTermMemory.scope_service,
    LongTermMemory.scope_env,
    LongTermMemory.content_hash,
)
_CORE_COLUMNS = (
    AgentCoreMemoryBlock.id,
    AgentCoreMemoryBlock.tenant_id,
    AgentCoreMemoryBlock.user_id,
    AgentCoreMemoryBlock.agent_id,
    AgentCoreMemoryBlock.block_key,
    AgentCoreMemoryBlock.description,
    AgentCoreMemoryBlock.content,
    AgentCoreMemoryBlock.max_tokens,
    AgentCoreMemoryBlock.version,
    AgentCoreMemoryBlock.read_only,
    AgentCoreMemoryBlock.source,
    AgentCoreMemoryBlock.content_hash,
    AgentCoreMemoryBlock.created_at,
    AgentCoreMemoryBlock.updated_at,
    AgentCoreMemoryBlock.status,
)


class PostgresMemoryRepository:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self.sessionmaker = sessionmaker

    async def ensure_ready(self) -> None:
        try:
            async with self.sessionmaker() as session:
                await self._probe_schema_capabilities(session)
        except MemoryPersistenceError:
            raise
        except OperationalError:
            raise MemoryStoreUnavailableError() from None
        except SQLAlchemyError:
            raise MemoryStoreContractError() from None

    async def ensure_default_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[CoreMemoryBlock]:
        _require_identity(tenant_id, user_id, agent_id)
        values = [
            {
                "id": str(uuid4()),
                "tenant_id": tenant_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "block_key": block_key,
                "description": CORE_BLOCK_SPECS[block_key].description,
                "content": "",
                "max_tokens": CORE_BLOCK_SPECS[block_key].max_tokens,
                "version": 1,
                "read_only": False,
                "source": "memory_service",
                "content_hash": "",
                "status": "active",
            }
            for block_key in DEFAULT_CORE_BLOCK_KEYS
        ]
        statement = insert(AgentCoreMemoryBlock).values(values).on_conflict_do_nothing(
            index_elements=[
                AgentCoreMemoryBlock.tenant_id,
                AgentCoreMemoryBlock.user_id,
                AgentCoreMemoryBlock.agent_id,
                AgentCoreMemoryBlock.block_key,
            ]
        )
        try:
            async with self.sessionmaker() as session:
                async with session.begin():
                    await session.execute(statement)
                    rows = (
                        await session.execute(
                            self._core_scope_statement(
                                tenant_id,
                                user_id,
                                agent_id,
                                block_keys=DEFAULT_CORE_BLOCK_KEYS,
                            ).with_for_update()
                        )
                    ).mappings().all()
                    blocks = [_core_record(row) for row in rows]
                    if (
                        len(blocks) != len(DEFAULT_CORE_BLOCK_KEYS)
                        or {block.block_key for block in blocks}
                        != set(DEFAULT_CORE_BLOCK_KEYS)
                        or any(block.status != "active" for block in blocks)
                    ):
                        raise CoreMemoryContractError()
            return _sort_core_blocks(blocks)
        except MemoryPersistenceError:
            raise
        except OperationalError:
            raise MemoryStoreUnavailableError() from None
        except SQLAlchemyError:
            raise MemoryStoreContractError() from None

    async def list_active_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[CoreMemoryBlock]:
        return await self._list_core_blocks(
            tenant_id,
            user_id,
            agent_id,
            statuses=("active",),
        )

    async def list_core_blocks_for_scope(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[CoreMemoryBlock]:
        return await self._list_core_blocks(
            tenant_id,
            user_id,
            agent_id,
            statuses=tuple(statuses) if statuses is not None else None,
        )

    async def _list_core_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        statuses: tuple[str, ...] | None,
    ) -> list[CoreMemoryBlock]:
        _require_identity(tenant_id, user_id, agent_id)
        _validate_statuses(statuses)
        statement = self._core_scope_statement(tenant_id, user_id, agent_id)
        if statuses is not None:
            statement = statement.where(AgentCoreMemoryBlock.status.in_(statuses))
        try:
            async with self.sessionmaker() as session:
                rows = (await session.execute(statement)).mappings().all()
                return _sort_core_blocks([_core_record(row) for row in rows])
        except MemoryPersistenceError:
            raise
        except OperationalError:
            raise MemoryStoreUnavailableError() from None
        except SQLAlchemyError:
            raise MemoryStoreContractError() from None

    async def cas_replace_core_content(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        block_key: str,
        content: str,
        *,
        expected_version: int,
    ) -> CoreContentWriteResult:
        _require_identity(tenant_id, user_id, agent_id)
        if block_key not in CORE_BLOCK_SPECS or expected_version < 1:
            raise MemoryStoreContractError()
        target_hash = content_hash(content)
        statement = (
            update(AgentCoreMemoryBlock)
            .where(
                AgentCoreMemoryBlock.tenant_id == tenant_id,
                AgentCoreMemoryBlock.user_id == user_id,
                AgentCoreMemoryBlock.agent_id == agent_id,
                AgentCoreMemoryBlock.block_key == block_key,
                AgentCoreMemoryBlock.status == "active",
                AgentCoreMemoryBlock.read_only.is_(False),
                AgentCoreMemoryBlock.version == expected_version,
                AgentCoreMemoryBlock.content_hash != target_hash,
            )
            .values(
                content=content,
                content_hash=target_hash,
                version=AgentCoreMemoryBlock.version + 1,
                updated_at=func.current_timestamp(),
            )
            .returning(*_CORE_COLUMNS)
        )
        try:
            async with self.sessionmaker() as session:
                async with session.begin():
                    row = (await session.execute(statement)).mappings().one_or_none()
                    if row is not None:
                        result = CoreContentWriteResult("updated", _core_record(row))
                    else:
                        current_row = (
                            await session.execute(
                                self._core_scope_statement(
                                    tenant_id,
                                    user_id,
                                    agent_id,
                                    block_keys=(block_key,),
                                )
                            )
                        ).mappings().one_or_none()
                        result = _classify_core_cas_miss(current_row, target_hash)
            return result
        except MemoryPersistenceError:
            raise
        except OperationalError:
            raise MemoryStoreUnavailableError() from None
        except SQLAlchemyError:
            raise MemoryStoreContractError() from None

    async def write_archival_exact(
        self,
        memory: MemoryRecord,
    ) -> ExactMemoryWriteResult:
        candidate = _normalize_archival_candidate(memory)
        insert_statement = (
            insert(LongTermMemory)
            .values(**_memory_insert_values(candidate))
            .on_conflict_do_nothing(
                index_elements=_active_exact_conflict_target(),
                index_where=text(ACTIVE_EXACT_PREDICATE_SQL),
            )
            .returning(*_MEMORY_COLUMNS)
        )
        try:
            async with self.sessionmaker() as session:
                async with session.begin():
                    row = (await session.execute(insert_statement)).mappings().one_or_none()
                    if row is not None:
                        result = ExactMemoryWriteResult(
                            status="written",
                            memory=_memory_record(row),
                            metadata_merged=False,
                        )
                    else:
                        existing_row = (
                            await session.execute(
                                self._exact_memory_statement(candidate).with_for_update()
                            )
                        ).mappings().one_or_none()
                        if existing_row is None:
                            raise MemoryExactConflictUnresolvedError()
                        existing = _memory_record(existing_row)
                        merged_tags = stable_tag_union(existing.tags, candidate.tags)
                        metadata_merged = merged_tags != existing.tags
                        if metadata_merged:
                            updated_row = (
                                await session.execute(
                                    update(LongTermMemory)
                                    .where(
                                        LongTermMemory.id == existing.id,
                                        LongTermMemory.tenant_id == candidate.tenant_id,
                                        LongTermMemory.user_id == candidate.user_id,
                                        LongTermMemory.agent_id == candidate.agent_id,
                                        LongTermMemory.status == "active",
                                    )
                                    .values(
                                        tags=merged_tags,
                                        updated_at=func.current_timestamp(),
                                    )
                                    .returning(*_MEMORY_COLUMNS)
                                )
                            ).mappings().one_or_none()
                            if updated_row is None:
                                raise MemoryExactConflictUnresolvedError()
                            existing = _memory_record(updated_row)
                        result = ExactMemoryWriteResult(
                            status="duplicate_skipped",
                            memory=existing,
                            metadata_merged=metadata_merged,
                        )
            return result
        except MemoryPersistenceError:
            raise
        except IntegrityError:
            raise MemoryExactConflictUnresolvedError() from None
        except OperationalError:
            raise MemoryStoreUnavailableError() from None
        except SQLAlchemyError:
            raise MemoryStoreContractError() from None

    async def list_active_memories(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        types: list[str] | None = None,
        scope_service: str | None = None,
        scope_env: str | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryRecord]:
        normalized_service = _blank_to_none(scope_service)
        normalized_env = _blank_to_none(scope_env)
        statement = self._memory_scope_statement(tenant_id, user_id, agent_id).where(
            LongTermMemory.status == "active"
        )
        if types:
            _validate_memory_types(types)
            statement = statement.where(LongTermMemory.type.in_(types))
        if normalized_service is not None:
            statement = statement.where(LongTermMemory.scope_service == normalized_service)
        if normalized_env is not None:
            statement = statement.where(LongTermMemory.scope_env == normalized_env)
        memories = await self._read_memories(statement)
        tag_set = set(stable_tag_union([], tags or []))
        if tag_set:
            memories = [
                memory for memory in memories if any(tag in tag_set for tag in memory.tags)
            ]
        return memories

    async def list_memories_for_scope(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[MemoryRecord]:
        normalized_statuses = tuple(statuses) if statuses is not None else None
        _validate_statuses(normalized_statuses)
        statement = self._memory_scope_statement(tenant_id, user_id, agent_id)
        if normalized_statuses is not None:
            statement = statement.where(LongTermMemory.status.in_(normalized_statuses))
        return await self._read_memories(statement)

    async def _read_memories(self, statement: Any) -> list[MemoryRecord]:
        try:
            async with self.sessionmaker() as session:
                rows = (await session.execute(statement)).mappings().all()
                return [_memory_record(row) for row in rows]
        except MemoryPersistenceError:
            raise
        except OperationalError:
            raise MemoryStoreUnavailableError() from None
        except SQLAlchemyError:
            raise MemoryStoreContractError() from None

    async def mark_returned(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        memory_ids: list[str],
    ) -> None:
        _require_identity(tenant_id, user_id, agent_id)
        ids = tuple(dict.fromkeys(memory_ids))
        if not ids:
            return
        if len(ids) > 1000 or any(not value.strip() for value in ids):
            raise MemoryStoreContractError()
        statement = (
            update(LongTermMemory)
            .where(
                LongTermMemory.tenant_id == tenant_id,
                LongTermMemory.user_id == user_id,
                LongTermMemory.agent_id == agent_id,
                LongTermMemory.status == "active",
                LongTermMemory.id.in_(ids),
            )
            .values(
                usage_count=LongTermMemory.usage_count + 1,
                last_used_at=func.current_timestamp(),
                updated_at=func.current_timestamp(),
            )
        )
        try:
            async with self.sessionmaker() as session:
                async with session.begin():
                    await session.execute(statement)
        except OperationalError:
            raise MemoryStoreUnavailableError() from None
        except SQLAlchemyError:
            raise MemoryStoreContractError() from None

    @staticmethod
    def _core_scope_statement(
        tenant_id: str,
        user_id: str,
        agent_id: str,
        *,
        block_keys: Sequence[str] | None = None,
    ) -> Any:
        statement = select(*_CORE_COLUMNS).where(
            AgentCoreMemoryBlock.tenant_id == tenant_id,
            AgentCoreMemoryBlock.user_id == user_id,
            AgentCoreMemoryBlock.agent_id == agent_id,
        )
        if block_keys is not None:
            statement = statement.where(AgentCoreMemoryBlock.block_key.in_(block_keys))
        return statement.order_by(
            case(
                {key: index for index, key in enumerate(DEFAULT_CORE_BLOCK_KEYS)},
                value=AgentCoreMemoryBlock.block_key,
                else_=len(DEFAULT_CORE_BLOCK_KEYS),
            ),
            AgentCoreMemoryBlock.block_key,
        )

    @staticmethod
    def _memory_scope_statement(tenant_id: str, user_id: str, agent_id: str) -> Any:
        _require_identity(tenant_id, user_id, agent_id)
        return (
            select(*_MEMORY_COLUMNS)
            .where(
                LongTermMemory.tenant_id == tenant_id,
                LongTermMemory.user_id == user_id,
                LongTermMemory.agent_id == agent_id,
            )
            .order_by(LongTermMemory.created_at, LongTermMemory.id)
        )

    @staticmethod
    def _exact_memory_statement(candidate: MemoryRecord) -> Any:
        return select(*_MEMORY_COLUMNS).where(
            LongTermMemory.tenant_id == candidate.tenant_id,
            LongTermMemory.user_id == candidate.user_id,
            LongTermMemory.agent_id == candidate.agent_id,
            LongTermMemory.type == candidate.type,
            func.coalesce(LongTermMemory.scope_service, "")
            == (candidate.scope_service or ""),
            func.coalesce(LongTermMemory.scope_env, "") == (candidate.scope_env or ""),
            LongTermMemory.content_hash == candidate.content_hash,
            LongTermMemory.status == "active",
        )

    @staticmethod
    async def _probe_schema_capabilities(session: AsyncSession) -> None:
        constraint_rows = (
            await session.execute(
                text(
                    "SELECT rel.relname AS table_name, con.conname, con.contype, "
                    "pg_get_constraintdef(con.oid) AS definition "
                    "FROM pg_constraint con "
                    "JOIN pg_class rel ON rel.oid = con.conrelid "
                    "WHERE rel.relname IN ('long_term_memory', 'agent_core_memory_block')"
                )
            )
        ).mappings().all()
        constraints = {str(row["conname"]): row for row in constraint_rows}
        if not M_P2_REQUIRED_CHECKS.issubset(constraints) or CORE_UNIQUE_CONSTRAINT not in constraints:
            raise MemoryStoreContractError()
        _validate_constraint_definitions(constraints)

        index_definition = await session.scalar(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'long_term_memory' AND indexname = :index_name"
            ),
            {"index_name": ACTIVE_EXACT_INDEX},
        )
        if not isinstance(index_definition, str) or not _valid_exact_index(index_definition):
            raise MemoryStoreContractError()

        vector_type = await session.scalar(
            text(
                "SELECT format_type(att.atttypid, att.atttypmod) "
                "FROM pg_attribute att JOIN pg_class rel ON rel.oid = att.attrelid "
                "WHERE rel.relname = 'long_term_memory' AND att.attname = 'embedding' "
                "AND att.attnum > 0 AND NOT att.attisdropped"
            )
        )
        if str(vector_type).lower() != "vector(1024)":
            raise MemoryStoreContractError()


def _active_exact_conflict_target() -> list[Any]:
    return [
        LongTermMemory.tenant_id,
        LongTermMemory.user_id,
        LongTermMemory.agent_id,
        LongTermMemory.type,
        func.coalesce(LongTermMemory.scope_service, literal_column("''")),
        func.coalesce(LongTermMemory.scope_env, literal_column("''")),
        LongTermMemory.content_hash,
    ]


def _memory_insert_values(memory: MemoryRecord) -> dict[str, Any]:
    return {
        "id": memory.id,
        "tenant_id": memory.tenant_id,
        "user_id": memory.user_id,
        "agent_id": memory.agent_id,
        "session_id": memory.session_id,
        "type": memory.type,
        "topic": memory.topic,
        "content": memory.content,
        "source": memory.source,
        "embedding": _embedding_for_write(memory),
        "embedding_model": memory.embedding_model,
        "embedding_dimension": memory.embedding_dimension,
        "embedding_metric": memory.embedding_metric,
        "embedding_version": memory.embedding_version,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "usage_count": memory.usage_count,
        "last_used_at": memory.last_used_at,
        "status": memory.status,
        "archived_at": memory.archived_at,
        "archive_reason": memory.archive_reason,
        "tags": memory.tags,
        "scope_service": memory.scope_service,
        "scope_env": memory.scope_env,
        "content_hash": memory.content_hash,
    }


def _embedding_for_write(memory: MemoryRecord) -> list[float] | None:
    if memory.embedding_model == "local-deterministic":
        return None
    if memory.embedding_dimension != 1024 or len(memory.embedding) != 1024:
        raise MemoryStoreContractError()
    if any(not math.isfinite(value) for value in memory.embedding):
        raise MemoryStoreContractError()
    return list(memory.embedding)


def _normalize_archival_candidate(memory: MemoryRecord) -> MemoryRecord:
    _require_identity(memory.tenant_id, memory.user_id, memory.agent_id)
    if memory.status != "active" or memory.type not in {"experience", "knowledge"}:
        raise MemoryStoreContractError()
    if not memory.topic.strip() or not memory.content.strip():
        raise MemoryStoreContractError()
    if (
        not memory.embedding_model.strip()
        or memory.embedding_dimension < 1
        or memory.embedding_metric != "cosine"
        or not memory.embedding_version.strip()
    ):
        raise MemoryStoreContractError()
    canonical_hash = canonical_content_hash(memory.content)
    return MemoryRecord(
        **{
            **memory.__dict__,
            "scope_service": _blank_to_none(memory.scope_service),
            "scope_env": _blank_to_none(memory.scope_env),
            "tags": stable_tag_union([], memory.tags),
            "content_hash": canonical_hash,
        }
    )


def _core_record(row: Mapping[str, Any]) -> CoreMemoryBlock:
    try:
        block = CoreMemoryBlock(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            user_id=str(row["user_id"]),
            agent_id=str(row["agent_id"]),
            block_key=str(row["block_key"]),
            description=str(row["description"]),
            content=str(row["content"]),
            max_tokens=int(row["max_tokens"]),
            version=int(row["version"]),
            read_only=bool(row["read_only"]),
            source=str(row["source"]),
            content_hash=str(row["content_hash"]),
            created_at=_datetime(row["created_at"]),
            updated_at=_datetime(row["updated_at"]),
            status=str(row["status"]),
        )
    except (KeyError, TypeError, ValueError):
        raise MemoryStoreContractError() from None
    _require_identity(block.tenant_id, block.user_id, block.agent_id)
    if (
        block.block_key not in CORE_BLOCK_SPECS
        or block.status not in {"active", "archived"}
        or block.version < 1
        or block.max_tokens < 1
        or block.content_hash != content_hash(block.content)
    ):
        raise MemoryStoreContractError()
    return block


def _memory_record(row: Mapping[str, Any]) -> MemoryRecord:
    try:
        raw_tags = row["tags"]
        if not isinstance(raw_tags, list) or any(not isinstance(tag, str) for tag in raw_tags):
            raise MemoryStoreContractError()
        tags = stable_tag_union([], raw_tags)
        if tags != raw_tags:
            raise MemoryStoreContractError()
        scope_service = _validated_scope(row["scope_service"])
        scope_env = _validated_scope(row["scope_env"])
        memory = MemoryRecord(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            user_id=str(row["user_id"]),
            agent_id=str(row["agent_id"]),
            session_id=str(row["session_id"]) if row["session_id"] is not None else None,
            type=str(row["type"]),
            topic=str(row["topic"]),
            content=str(row["content"]),
            source=str(row["source"]),
            embedding=[],
            embedding_model=str(row["embedding_model"]),
            embedding_dimension=int(row["embedding_dimension"]),
            embedding_metric=str(row["embedding_metric"]),
            embedding_version=str(row["embedding_version"]),
            created_at=_datetime(row["created_at"]),
            updated_at=_datetime(row["updated_at"]),
            usage_count=int(row["usage_count"]),
            last_used_at=_optional_datetime(row["last_used_at"]),
            status=str(row["status"]),
            archived_at=_optional_datetime(row["archived_at"]),
            archive_reason=(
                str(row["archive_reason"]) if row["archive_reason"] is not None else None
            ),
            tags=tags,
            scope_service=scope_service,
            scope_env=scope_env,
            content_hash=str(row["content_hash"] or ""),
        )
    except MemoryPersistenceError:
        raise
    except (KeyError, TypeError, ValueError):
        raise MemoryStoreContractError() from None
    _require_identity(memory.tenant_id, memory.user_id, memory.agent_id)
    if (
        memory.type not in {"rule", "experience", "knowledge"}
        or memory.status not in {"active", "archived"}
        or memory.usage_count < 0
        or memory.embedding_dimension < 1
        or memory.embedding_metric != "cosine"
        or not memory.embedding_model.strip()
        or not memory.embedding_version.strip()
    ):
        raise MemoryStoreContractError()
    if memory.status == "active" and (
        not _SHA256_RE.fullmatch(memory.content_hash)
        or memory.content_hash != canonical_content_hash(memory.content)
    ):
        raise MemoryStoreContractError()
    return memory


def _classify_core_cas_miss(
    row: Mapping[str, Any] | None,
    target_hash: str,
) -> CoreContentWriteResult:
    if row is None:
        return CoreContentWriteResult("inactive", None)
    block = _core_record(row)
    if block.status != "active":
        return CoreContentWriteResult("inactive", block)
    if block.read_only:
        return CoreContentWriteResult("read_only", block)
    if block.content_hash == target_hash:
        return CoreContentWriteResult("unchanged", block)
    return CoreContentWriteResult("conflict", block)


def _sort_core_blocks(blocks: list[CoreMemoryBlock]) -> list[CoreMemoryBlock]:
    order = {key: index for index, key in enumerate(DEFAULT_CORE_BLOCK_KEYS)}
    return sorted(blocks, key=lambda block: (order.get(block.block_key, len(order)), block.block_key))


def _require_identity(tenant_id: str, user_id: str, agent_id: str) -> None:
    if any(not value or not value.strip() for value in (tenant_id, user_id, agent_id)):
        raise MemoryStoreContractError()


def _validate_statuses(statuses: tuple[str, ...] | None) -> None:
    if statuses is not None and any(status not in {"active", "archived"} for status in statuses):
        raise MemoryStoreContractError()


def _validate_memory_types(types: Sequence[str]) -> None:
    if any(memory_type not in {"rule", "experience", "knowledge"} for memory_type in types):
        raise MemoryStoreContractError()


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None


def _validated_scope(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _blank_to_none(value) != value:
        raise MemoryStoreContractError()
    return value


def _datetime(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise MemoryStoreContractError()
    return value


def _optional_datetime(value: Any) -> datetime | None:
    return None if value is None else _datetime(value)


def _normalized_definition(value: Any) -> str:
    return " ".join(str(value).replace('"', "").lower().split())


def _validate_constraint_definitions(constraints: Mapping[str, Mapping[str, Any]]) -> None:
    required_fragments = {
        "chk_long_term_memory_active_content_hash": ("status", "active", "content_hash", "64"),
        "chk_long_term_memory_scope_service_nonblank": ("scope_service", "btrim"),
        "chk_long_term_memory_scope_env_nonblank": ("scope_env", "btrim"),
        "chk_long_term_memory_tags_array": ("jsonb_typeof", "tags", "array"),
        "chk_core_memory_block_key": (
            "block_key",
            "user_rules",
            "user_ops_profile",
            "service_notes",
        ),
        "chk_core_memory_version_positive": ("version", "1"),
        "chk_core_memory_max_tokens_positive": ("max_tokens", "0"),
        "chk_core_memory_content_hash_format": ("content_hash", "64"),
    }
    for name, fragments in required_fragments.items():
        definition = _normalized_definition(constraints[name]["definition"])
        if constraints[name]["contype"] != "c" or any(
            fragment not in definition for fragment in fragments
        ):
            raise MemoryStoreContractError()
    core_unique = constraints[CORE_UNIQUE_CONSTRAINT]
    unique_definition = _normalized_definition(core_unique["definition"])
    if core_unique["contype"] != "u" or not all(
        fragment in unique_definition
        for fragment in ("tenant_id", "user_id", "agent_id", "block_key")
    ):
        raise MemoryStoreContractError()


def _valid_exact_index(definition: str) -> bool:
    normalized = _normalized_definition(definition)
    key_fragments = (
        "unique index",
        "tenant_id",
        "user_id",
        "agent_id",
        "type",
        "coalesce(scope_service",
        "coalesce(scope_env",
        "content_hash",
    )
    positions = [normalized.find(fragment) for fragment in key_fragments]
    if not all(position >= 0 for position in positions) or positions != sorted(positions):
        return False
    where_position = normalized.find("where", positions[-1])
    status_position = normalized.find("status", where_position)
    active_position = normalized.find("active", status_position)
    return positions[-1] < where_position < status_position < active_position
