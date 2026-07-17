from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import sys

from superbiz_agent.config import Settings
from superbiz_agent.memory.backfill import MemoryEmbeddingBackfillService
from superbiz_agent.memory.embedding import build_memory_embedding_service
from superbiz_agent.memory.errors import MemoryPersistenceError
from superbiz_agent.persistence.database import create_engine, create_sessionmaker
from superbiz_agent.persistence.repositories.memory import PostgresMemoryRepository


async def _run(*, dry_run: bool, batch_size: int | None) -> int:
    engine = None
    embedding_service = None
    try:
        settings = Settings()
        if settings.memory_store_backend != "postgres":
            raise ValueError("memory_store_backend must be postgres")
        engine = create_engine(settings.database_url)
        repository = PostgresMemoryRepository(create_sessionmaker(engine))
        embedding_service = build_memory_embedding_service(settings)
        await repository.ensure_ready()
        report = await MemoryEmbeddingBackfillService(
            repository,
            embedding_service,
            batch_size=batch_size or settings.memory_embedding_batch_size,
        ).run(dry_run=dry_run)
        print(json.dumps(asdict(report), sort_keys=True))
        return 0 if report.status == "passed" else 1
    except MemoryPersistenceError as exc:
        print(json.dumps({"status": "failed", "error_codes": [exc.code]}, sort_keys=True))
        return 1
    except Exception:
        print(
            json.dumps(
                {"status": "failed", "error_codes": ["memory_backfill_configuration_error"]},
                sort_keys=True,
            )
        )
        return 1
    finally:
        if embedding_service is not None:
            try:
                await embedding_service.aclose()
            except Exception:
                pass
        if engine is not None:
            await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill active archival memory embeddings")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    return asyncio.run(_run(dry_run=args.dry_run, batch_size=args.batch_size))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
