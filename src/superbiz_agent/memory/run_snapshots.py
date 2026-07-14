from __future__ import annotations

import threading
from dataclasses import dataclass

from superbiz_agent.harness.context import RunContext
from superbiz_agent.memory.errors import MemoryStoreIsolationError
from superbiz_agent.memory.schemas import CoreMemoryBlock


@dataclass
class _RunSnapshot:
    identity: tuple[str, str, str]
    versions: dict[str, int]


class CoreVersionSnapshotRegistry:
    """Process-local trusted Core versions captured from model-visible context."""

    def __init__(self) -> None:
        self._snapshots: dict[str, _RunSnapshot] = {}
        self._lock = threading.Lock()

    def capture_once(self, run_context: RunContext, blocks: list[CoreMemoryBlock]) -> None:
        identity = _identity(run_context)
        versions = {block.block_key: block.version for block in blocks}
        with self._lock:
            existing = self._snapshots.get(run_context.run_id)
            if existing is None:
                self._snapshots[run_context.run_id] = _RunSnapshot(identity, versions)
                return
            if existing.identity != identity:
                raise MemoryStoreIsolationError()

    def expected_version(self, run_context: RunContext, block_key: str) -> int | None:
        identity = _identity(run_context)
        with self._lock:
            snapshot = self._snapshots.get(run_context.run_id)
            if snapshot is None:
                return None
            if snapshot.identity != identity:
                raise MemoryStoreIsolationError()
            return snapshot.versions.get(block_key)

    def advance(self, run_context: RunContext, block_key: str, version: int) -> None:
        identity = _identity(run_context)
        with self._lock:
            snapshot = self._snapshots.get(run_context.run_id)
            if snapshot is None or snapshot.identity != identity:
                raise MemoryStoreIsolationError()
            snapshot.versions[block_key] = version

    def cleanup(self, run_id: str) -> None:
        with self._lock:
            self._snapshots.pop(run_id, None)

    def contains(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._snapshots


def _identity(run_context: RunContext) -> tuple[str, str, str]:
    context = run_context.request_context
    return (
        context.tenant_id or "",
        context.user_id or "",
        context.agent_id or "",
    )

