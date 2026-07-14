from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from superbiz_agent.persistence.models import RagDocument, RagKnowledgeBase
from superbiz_agent.persistence.repositories.rag import (
    RagClaimLostError,
    RagDocumentArchivedError,
    RagDocumentClaimStatus,
    RagDocumentRepository,
    RagKnowledgeBaseNotActiveError,
    RagKnowledgeBaseContractError,
    RagKnowledgeBaseRepository,
    RagPersistenceError,
)


NOW = datetime(2026, 7, 12, 2, 0, tzinfo=timezone.utc)


def _compiled(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


class _Result:
    def __init__(self, row=None, *, rows=None, rowcount: int = 0) -> None:
        self.row = row
        self.rows = list(rows) if rows is not None else None
        self.rowcount = rowcount

    def scalar_one_or_none(self):
        return self.row

    def scalars(self):
        return self

    def all(self):
        if self.rows is not None:
            return self.rows
        return [] if self.row is None else [self.row]


class _Session:
    def __init__(self, *, scalars=(), results=(), commit_error: Exception | None = None) -> None:
        self.scalars = list(scalars)
        self.results = list(results)
        self.commit_error = commit_error
        self.statements = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def scalar(self, statement):
        self.statements.append(statement)
        return self.scalars.pop(0)

    async def execute(self, statement):
        self.statements.append(statement)
        return self.results.pop(0)

    async def commit(self) -> None:
        if self.commit_error is not None:
            raise self.commit_error
        self.commits += 1


class _SessionFactory:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __call__(self) -> _Session:
        return self.session


def _kb(*, tenant_id: str = "tenant-a", status: str = "active") -> RagKnowledgeBase:
    return RagKnowledgeBase(
        id="kb-a",
        tenant_id=tenant_id,
        name="Default",
        description=None,
        status=status,
        is_default=True,
        created_by="admin",
        created_at=NOW,
        updated_at=NOW,
    )


def _document(
    *,
    status: str = "indexing",
    document_id: str = "document-existing",
    claim_token: str | None = "old-token",
    claimed_at: datetime | None = NOW,
    chunk_count: int = 0,
) -> RagDocument:
    if status != "indexing":
        claim_token = None
        claimed_at = None
    return RagDocument(
        id=document_id,
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        source_uri="trusted://runbook",
        document_name="runbook.md",
        content_type="markdown",
        parser="batch-b",
        parser_version="1",
        embedding_provider="stub",
        embedding_model="embedding-test",
        embedding_version="v1",
        embedding_dimension=4,
        status=status,
        content_hash="a" * 64,
        chunk_count=chunk_count,
        claim_token=claim_token,
        claimed_at=claimed_at,
        failure_code="old_failure" if status == "failed" else None,
        failure_message="old safe message" if status == "failed" else None,
        created_by="admin",
        created_at=NOW,
        updated_at=NOW,
    )


def _claim_kwargs() -> dict[str, object]:
    return {
        "tenant_id": "tenant-a",
        "knowledge_base_id": "kb-a",
        "source_uri": "trusted://runbook",
        "document_name": "runbook.md",
        "content_type": "markdown",
        "parser": "batch-b",
        "parser_version": "1",
        "embedding_provider": "stub",
        "embedding_model": "embedding-test",
        "embedding_version": "v1",
        "embedding_dimension": 4,
        "content_hash": "a" * 64,
        "created_by": "admin",
        "stale_after_seconds": 900,
    }


def test_models_freeze_tenant_fk_partial_default_and_state_constraints() -> None:
    kb_table = RagKnowledgeBase.__table__
    document_table = RagDocument.__table__
    kb_constraints = {constraint.name: constraint for constraint in kb_table.constraints}
    document_constraints = {
        constraint.name: constraint for constraint in document_table.constraints
    }
    indexes = {index.name: index for index in kb_table.indexes}

    tenant_identity = kb_constraints["uq_rag_knowledge_base_tenant_id_id"]
    assert [column.name for column in tenant_identity.columns] == ["tenant_id", "id"]
    default_index = indexes["uq_rag_knowledge_base_active_default"]
    assert default_index.unique is True
    assert "status = 'active' AND is_default = TRUE" in str(
        default_index.dialect_options["postgresql"]["where"]
    )

    foreign_key = document_constraints["fk_rag_document_tenant_knowledge_base"]
    assert [element.parent.name for element in foreign_key.elements] == [
        "tenant_id",
        "knowledge_base_id",
    ]
    assert [element.target_fullname for element in foreign_key.elements] == [
        "rag_knowledge_base.tenant_id",
        "rag_knowledge_base.id",
    ]
    assert "claim_token IS NOT NULL" in str(
        document_constraints["chk_rag_document_claim_state"].sqltext
    )
    assert "claim_token IS NULL" in str(
        document_constraints["chk_rag_document_claim_state"].sqltext
    )
    assert "chunk_count >= 0" == str(document_constraints["chk_rag_document_chunk_count"].sqltext)
    assert "content_hash ~ '^[0-9a-f]{64}$'" == str(
        document_constraints["chk_rag_document_content_hash_sha256"].sqltext
    )
    scope_hash = document_constraints["uq_rag_document_scope_content_hash"]
    assert [column.name for column in scope_hash.columns] == [
        "tenant_id",
        "knowledge_base_id",
        "content_hash",
    ]


def test_repository_statements_always_apply_tenant_and_fencing_filters() -> None:
    active_sql = _compiled(
        RagKnowledgeBaseRepository.active_statement(tenant_id="tenant-a", knowledge_base_id="kb-a")
    )
    existing_sql = _compiled(
        RagDocumentRepository.existing_statement(
            tenant_id="tenant-a", knowledge_base_id="kb-a", content_hash="hash-a"
        )
    )
    refresh_sql = _compiled(
        RagDocumentRepository.refresh_claim_statement(
            tenant_id="tenant-a", document_id="document-a", claim_token="token-a"
        )
    )
    active_update_sql = _compiled(
        RagDocumentRepository.mark_active_statement(
            tenant_id="tenant-a",
            document_id="document-a",
            claim_token="token-a",
            chunk_count=3,
        )
    )
    statuses_sql = _compiled(
        RagDocumentRepository.statuses_statement(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            document_ids=("document-a", "document-b"),
        )
    )

    assert "rag_knowledge_base.tenant_id = 'tenant-a'" in active_sql
    assert "rag_knowledge_base.status = 'active'" in active_sql
    assert "rag_document.tenant_id = 'tenant-a'" in existing_sql
    assert "rag_document.knowledge_base_id = 'kb-a'" in existing_sql
    assert "rag_document.tenant_id = 'tenant-a'" in statuses_sql
    assert "rag_document.knowledge_base_id = 'kb-a'" in statuses_sql
    assert "rag_document.id IN ('document-a', 'document-b')" in statuses_sql
    for sql in (refresh_sql, active_update_sql):
        assert "rag_document.tenant_id = 'tenant-a'" in sql
        assert "rag_document.id = 'document-a'" in sql
        assert "rag_document.status = 'indexing'" in sql
        assert "rag_document.claim_token = 'token-a'" in sql
    assert "claimed_at=CURRENT_TIMESTAMP" in refresh_sql
    assert "claim_token=NULL" in active_update_sql
    assert "claimed_at=NULL" in active_update_sql


@pytest.mark.asyncio
async def test_first_claim_uses_on_conflict_and_commits_before_return() -> None:
    inserted = _document(document_id="document-new", claim_token="new-token")
    session = _Session(scalars=[_kb()], results=[_Result(inserted)])
    repository = RagDocumentRepository(_SessionFactory(session))

    claim = await repository.claim_for_ingestion(**_claim_kwargs())

    assert claim.status is RagDocumentClaimStatus.CLAIMED
    assert claim.document_id == "document-new"
    assert claim.claim_token == "new-token"
    assert session.commits == 1
    insert_sql = _compiled(session.statements[1])
    assert "ON CONFLICT (tenant_id, knowledge_base_id, content_hash) DO NOTHING" in insert_sql
    assert "CURRENT_TIMESTAMP" in insert_sql
    assert "RETURNING rag_document" in insert_sql
    assert "FOR UPDATE" in _compiled(session.statements[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_content_hash",
    [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
    ],
    ids=["short", "too-long", "uppercase", "non-hex"],
)
async def test_claim_rejects_noncanonical_sha256_content_hash(
    invalid_content_hash: str,
) -> None:
    session = _Session()
    repository = RagDocumentRepository(_SessionFactory(session))
    kwargs = _claim_kwargs()
    kwargs["content_hash"] = invalid_content_hash

    with pytest.raises(ValueError, match="64-character lowercase hexadecimal SHA-256"):
        await repository.claim_for_ingestion(**kwargs)

    assert session.statements == []
    assert session.commits == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row", "expected_status"),
    [
        (_document(status="active", chunk_count=7), RagDocumentClaimStatus.DUPLICATE_SKIPPED),
        (_document(claimed_at=NOW - timedelta(seconds=899)), RagDocumentClaimStatus.IN_PROGRESS),
    ],
)
async def test_existing_active_or_fresh_claim_has_no_new_side_effect(
    row: RagDocument, expected_status: RagDocumentClaimStatus
) -> None:
    scalar_values = [_kb(), row]
    if row.status == "indexing":
        scalar_values.append(NOW)
    session = _Session(scalars=scalar_values, results=[_Result(None)])
    repository = RagDocumentRepository(_SessionFactory(session))

    claim = await repository.claim_for_ingestion(**_claim_kwargs())

    assert claim.status is expected_status
    assert claim.document_id == "document-existing"
    assert claim.claim_token is None
    assert claim.chunk_count == row.chunk_count
    assert session.commits == 1
    assert "FOR UPDATE" in _compiled(session.statements[2])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        _document(status="failed", chunk_count=5),
        _document(claimed_at=NOW - timedelta(seconds=901), chunk_count=5),
    ],
)
async def test_failed_and_stale_claim_reuse_row_and_rotate_fencing_token(row: RagDocument) -> None:
    scalar_values = [_kb(), row]
    if row.status == "indexing":
        scalar_values.append(NOW)
    session = _Session(scalars=scalar_values, results=[_Result(None)])
    repository = RagDocumentRepository(_SessionFactory(session))

    claim = await repository.claim_for_ingestion(**_claim_kwargs())

    assert claim.status is RagDocumentClaimStatus.CLAIMED
    assert claim.document_id == "document-existing"
    assert claim.claim_token not in (None, "old-token")
    assert row.status == "indexing"
    assert row.chunk_count == 0
    assert row.failure_code is None
    assert row.failure_message is None
    assert session.commits == 1


@pytest.mark.asyncio
async def test_claim_rejects_inactive_kb_before_document_insert() -> None:
    session = _Session(scalars=[None])
    repository = RagDocumentRepository(_SessionFactory(session))

    with pytest.raises(RagKnowledgeBaseNotActiveError) as exc_info:
        await repository.claim_for_ingestion(**_claim_kwargs())

    assert exc_info.value.code == "knowledge_base_not_active"
    assert len(session.statements) == 1
    assert session.commits == 0


@pytest.mark.asyncio
async def test_archived_document_is_never_reactivated() -> None:
    row = _document(status="archived", chunk_count=4)
    session = _Session(scalars=[_kb(), row], results=[_Result(None)])
    repository = RagDocumentRepository(_SessionFactory(session))

    with pytest.raises(RagDocumentArchivedError):
        await repository.claim_for_ingestion(**_claim_kwargs())

    assert row.status == "archived"
    assert row.chunk_count == 4
    assert session.commits == 0


@pytest.mark.asyncio
async def test_claim_commit_failure_never_returns_claim() -> None:
    session = _Session(
        scalars=[_kb()],
        results=[_Result(_document())],
        commit_error=SQLAlchemyError("commit failed"),
    )
    repository = RagDocumentRepository(_SessionFactory(session))

    with pytest.raises(RagPersistenceError) as exc_info:
        await repository.claim_for_ingestion(**_claim_kwargs())

    assert exc_info.value.code == "document_store_error"


@pytest.mark.asyncio
async def test_fenced_updates_commit_once_and_claim_loss_does_not_commit() -> None:
    success_session = _Session(results=[_Result(rowcount=1)])
    repository = RagDocumentRepository(_SessionFactory(success_session))
    await repository.refresh_claim(
        tenant_id="tenant-a", document_id="document-a", claim_token="token-a"
    )
    assert success_session.commits == 1

    lost_session = _Session(results=[_Result(rowcount=0)])
    repository = RagDocumentRepository(_SessionFactory(lost_session))
    with pytest.raises(RagClaimLostError) as exc_info:
        await repository.mark_active(
            tenant_id="tenant-a",
            document_id="document-a",
            claim_token="stale-token",
            chunk_count=2,
        )
    assert exc_info.value.code == "claim_lost"
    assert lost_session.commits == 0


@pytest.mark.asyncio
async def test_mark_failed_bounds_safe_metadata_and_clears_claim() -> None:
    session = _Session(results=[_Result(rowcount=1)])
    repository = RagDocumentRepository(_SessionFactory(session))

    await repository.mark_failed(
        tenant_id="tenant-a",
        document_id="document-a",
        claim_token="token-a",
        failure_code="x" * 120,
        sanitized_message="m" * 1200,
    )

    statement = session.statements[0]
    params = statement.compile(dialect=postgresql.dialect()).params
    assert len(params["failure_code"]) == 100
    assert len(params["failure_message"]) == 1000
    sql = _compiled(statement)
    assert "claim_token=NULL" in sql
    assert "claimed_at=NULL" in sql
    assert session.commits == 1


@pytest.mark.asyncio
async def test_create_default_conflict_reads_existing_and_commits() -> None:
    existing = _kb()
    session = _Session(scalars=[existing], results=[_Result(None)])
    repository = RagKnowledgeBaseRepository(_SessionFactory(session))

    record = await repository.create_default(
        tenant_id="tenant-a", name="Ignored concurrent name", created_by="admin"
    )

    assert record.id == "kb-a"
    assert session.commits == 1
    insert_sql = _compiled(session.statements[0])
    assert "ON CONFLICT (tenant_id)" in insert_sql
    assert "WHERE status = 'active' AND is_default = TRUE" in insert_sql


@pytest.mark.asyncio
async def test_default_kb_read_limits_two_and_fails_closed_on_duplicates() -> None:
    duplicate_session = _Session(results=[_Result(rows=[_kb(), _kb()])])
    repository = RagKnowledgeBaseRepository(_SessionFactory(duplicate_session))

    with pytest.raises(RagKnowledgeBaseContractError):
        await repository.get_default_for_tenant(tenant_id="tenant-a")

    assert "LIMIT 2" in _compiled(duplicate_session.statements[0])


@pytest.mark.asyncio
async def test_document_statuses_are_scoped_deduplicated_and_empty_is_zero_io() -> None:
    session = _Session(results=[_Result(rows=[("document-a", "active"), ("document-b", "failed")])])
    repository = RagDocumentRepository(_SessionFactory(session))

    statuses = await repository.get_document_statuses(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_ids=("document-a", "document-b", "document-a"),
    )

    assert statuses == {"document-a": "active", "document-b": "failed"}
    sql = _compiled(session.statements[0])
    assert "rag_document.tenant_id = 'tenant-a'" in sql
    assert "rag_document.knowledge_base_id = 'kb-a'" in sql
    assert "rag_document.id IN ('document-a', 'document-b')" in sql

    empty_session = _Session()
    empty_repository = RagDocumentRepository(_SessionFactory(empty_session))
    assert await empty_repository.get_document_statuses(
        tenant_id="tenant-a", knowledge_base_id="kb-a", document_ids=()
    ) == {}
    assert empty_session.statements == []


@pytest.mark.asyncio
async def test_document_status_repository_rejects_more_than_frozen_top_k() -> None:
    repository = RagDocumentRepository(_SessionFactory(_Session()))
    with pytest.raises(ValueError, match="safety limit"):
        await repository.get_document_statuses(
            tenant_id="tenant-a",
            knowledge_base_id="kb-a",
            document_ids=tuple(f"document-{index}" for index in range(1001)),
        )


def test_migration_upgrade_downgrade_are_symmetric() -> None:
    migration_path = (
        Path(__file__).parents[1] / "alembic" / "versions" / "20260712_01_create_rag_metadata.py"
    )
    spec = spec_from_file_location("rag_metadata_migration", migration_path)
    assert spec is not None and spec.loader is not None
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)

    class RecordingOp:
        def __init__(self) -> None:
            self.created_tables = []
            self.dropped_tables = []
            self.created_indexes = []
            self.dropped_indexes = []

        def create_table(self, name, *args, **kwargs):
            self.created_tables.append((name, args, kwargs))

        def drop_table(self, name, *args, **kwargs):
            self.dropped_tables.append(name)

        def create_index(self, name, *args, **kwargs):
            self.created_indexes.append((name, args, kwargs))

        def drop_index(self, name, *args, **kwargs):
            self.dropped_indexes.append(name)

    recorder = RecordingOp()
    migration.op = recorder
    migration.upgrade()
    migration.downgrade()

    assert migration.down_revision == "20260705_02"
    assert [table[0] for table in recorder.created_tables] == [
        "rag_knowledge_base",
        "rag_document",
    ]
    assert recorder.dropped_tables == ["rag_document", "rag_knowledge_base"]
    assert {index[0] for index in recorder.created_indexes} == set(recorder.dropped_indexes)
    kb_table_args = recorder.created_tables[0][1]
    partial_index = next(
        index
        for index in recorder.created_indexes
        if index[0] == "uq_rag_knowledge_base_active_default"
    )
    assert partial_index[2]["unique"] is True
    assert "status = 'active' AND is_default = TRUE" in str(partial_index[2]["postgresql_where"])
    assert any(
        getattr(constraint, "name", None) == "uq_rag_knowledge_base_tenant_id_id"
        for constraint in kb_table_args
    )
    document_table_args = recorder.created_tables[1][1]
    assert any(
        getattr(constraint, "name", None) == "fk_rag_document_tenant_knowledge_base"
        for constraint in document_table_args
    )
    content_hash_constraint = next(
        constraint
        for constraint in document_table_args
        if getattr(constraint, "name", None) == "chk_rag_document_content_hash_sha256"
    )
    assert str(content_hash_constraint.sqltext) == "content_hash ~ '^[0-9a-f]{64}$'"
