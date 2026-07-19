from __future__ import annotations

import json
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from superbiz_agent.persistence.alembic_url import (
    CONNECTION_VALIDATOR_ATTRIBUTE,
    EXPLICIT_DATABASE_URL_ATTRIBUTE,
)


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_TEST = ROOT / "tests" / "test_memory_retrieval_postgres_acceptance.py"
PENDING_EXIT_CODE = 3
CONFIRMATION_VALUE = "ERASE_M_P1_ISOLATED_TEST_DATABASE"
DEDICATED_MARKERS = frozenset(
    {
        "superbiz-agent:m-p2-destructive-test-database:v1",
        "superbiz-agent:m-p1-retrieval-test-database:v1",
    }
)
FORBIDDEN_DATABASES = frozenset(
    {"postgres", "template0", "template1", "super_biz_agent", "superbiz_agent"}
)


def _upgrade_dedicated_database(database_url: str) -> None:
    parsed = make_url(database_url)
    database = (parsed.database or "").strip()
    if (
        os.environ.get("M_P1_TEST_DATABASE_DESTRUCTIVE_CONFIRM") != CONFIRMATION_VALUE
        or parsed.get_backend_name() != "postgresql"
        or parsed.get_driver_name() != "asyncpg"
        or not parsed.host
        or not database
        or database.lower() in FORBIDDEN_DATABASES
        or ("m_p1_test" not in database.lower() and "m_p2_test" not in database.lower())
    ):
        raise RuntimeError("M-P1 acceptance database is not dedicated-test-only.")
    config = Config()
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.attributes[EXPLICIT_DATABASE_URL_ATTRIBUTE] = database_url

    def validate_connection(connection) -> None:
        actual_database = connection.scalar(text("SELECT current_database()"))
        marker = connection.scalar(
            text(
                "SELECT shobj_description(oid, 'pg_database') FROM pg_database "
                "WHERE datname = current_database()"
            )
        )
        if actual_database != database or marker not in DEDICATED_MARKERS:
            raise RuntimeError("M-P1 acceptance database marker is invalid.")

    config.attributes[CONNECTION_VALIDATOR_ATTRIBUTE] = validate_connection
    command.upgrade(config, "head")


class _AcceptanceStats:
    def __init__(self) -> None:
        self.planned = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def pytest_collection_finish(self, session) -> None:
        self.planned = len(session.items)

    def pytest_runtest_logreport(self, report) -> None:
        if report.when != "call":
            return
        if report.passed:
            self.passed += 1
        elif report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1


def main() -> int:
    database_url = os.environ.get("M_P1_TEST_DATABASE_URL")
    if not database_url:
        print(
            json.dumps(
                {
                    "status": "pending",
                    "reason": "M_P1_TEST_DATABASE_URL_missing",
                    "exit_code": PENDING_EXIT_CODE,
                },
                sort_keys=True,
            )
        )
        return PENDING_EXIT_CODE
    try:
        _upgrade_dedicated_database(database_url)
    except Exception:
        print(
            json.dumps(
                {"status": "failed", "error_codes": ["memory_retrieval_migration_error"]},
                sort_keys=True,
            )
        )
        return 1
    stats = _AcceptanceStats()
    pytest_exit = int(pytest.main(["-q", str(ACCEPTANCE_TEST), "-rs"], plugins=[stats]))
    complete = (
        pytest_exit == 0
        and stats.planned > 0
        and stats.passed == stats.planned
        and stats.failed == 0
        and stats.skipped == 0
    )
    print(
        json.dumps(
            {
                "status": "passed" if complete else "failed",
                "planned": stats.planned,
                "executed": stats.passed + stats.failed,
                "skipped": stats.skipped,
                "passed": stats.passed,
                "failed": stats.failed,
            },
            sort_keys=True,
        )
    )
    return 0 if complete else (pytest_exit or 1)


if __name__ == "__main__":
    raise SystemExit(main())
