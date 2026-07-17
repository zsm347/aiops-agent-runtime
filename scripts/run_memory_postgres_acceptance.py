from __future__ import annotations

import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_TEST = ROOT / "tests" / "test_memory_postgres_acceptance.py"
PENDING_EXIT_CODE = 3


def main() -> int:
    if not os.environ.get("M_P2_TEST_DATABASE_URL"):
        print(
            "status=pending reason=M_P2_TEST_DATABASE_URL_missing "
            f"exit_code={PENDING_EXIT_CODE}"
        )
        return PENDING_EXIT_CODE

    exit_code = int(pytest.main(["-q", str(ACCEPTANCE_TEST), "-rs"]))
    status = "passed" if exit_code == 0 else "failed"
    print(f"status={status} exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
