from __future__ import annotations

from dataclasses import dataclass

from superbiz_agent.memory.embedding import (
    REAL_MEMORY_EMBEDDING_PROVIDERS,
    MemoryEmbeddingService,
)
from superbiz_agent.memory.errors import MemoryPersistenceError, MemoryStoreContractError
from superbiz_agent.memory.ports import MemoryEmbeddingBackfillRepository


@dataclass(frozen=True)
class MemoryEmbeddingBackfillReport:
    status: str
    dry_run: bool
    eligible: int
    scanned: int
    embedded: int
    updated: int
    concurrent_skipped: int
    failed: int
    batches: int
    error_codes: tuple[str, ...] = ()


class MemoryEmbeddingBackfillService:
    def __init__(
        self,
        repository: MemoryEmbeddingBackfillRepository,
        embedding_service: MemoryEmbeddingService,
        *,
        batch_size: int,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
            or batch_size > 1000
            or embedding_service.identity.provider not in REAL_MEMORY_EMBEDDING_PROVIDERS
            or embedding_service.identity.dimension != 1024
        ):
            raise MemoryStoreContractError()
        self.repository = repository
        self.embedding_service = embedding_service
        self.batch_size = batch_size

    async def run(self, *, dry_run: bool = False) -> MemoryEmbeddingBackfillReport:
        if not isinstance(dry_run, bool):
            raise MemoryStoreContractError()
        identity = self.embedding_service.identity
        eligible = await self.repository.count_embedding_backfill_candidates(identity)
        scanned = embedded = updated = concurrent_skipped = failed = batches = 0
        after_id: str | None = None
        error_codes: list[str] = []

        while True:
            candidates = await self.repository.list_embedding_backfill_candidates(
                identity,
                after_id=after_id,
                limit=self.batch_size,
            )
            if not candidates:
                break
            batches += 1
            scanned += len(candidates)
            after_id = candidates[-1].id
            if dry_run:
                continue

            try:
                batch = await self.embedding_service.embed_documents(
                    tuple(candidate.content for candidate in candidates)
                )
                if batch.identity != identity or len(batch.vectors) != len(candidates):
                    raise MemoryStoreContractError()
                embedded += len(batch.vectors)
                for candidate, vector in zip(candidates, batch.vectors, strict=True):
                    if await self.repository.apply_embedding_backfill(
                        candidate,
                        vector,
                        identity,
                    ):
                        updated += 1
                    else:
                        concurrent_skipped += 1
            except MemoryPersistenceError as exc:
                failed += len(candidates)
                error_codes.append(exc.code)
                break

        status = "passed" if failed == 0 else "failed"
        return MemoryEmbeddingBackfillReport(
            status=status,
            dry_run=dry_run,
            eligible=eligible,
            scanned=scanned,
            embedded=embedded,
            updated=updated,
            concurrent_skipped=concurrent_skipped,
            failed=failed,
            batches=batches,
            error_codes=tuple(dict.fromkeys(error_codes)),
        )
