from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from xml.sax.saxutils import escape, quoteattr

from superbiz_agent.harness.context import RunContext
from superbiz_agent.memory.policy import MemoryWritePolicy, PolicyDecision
from superbiz_agent.memory.ports import MemoryRepository
from superbiz_agent.memory.run_snapshots import CoreVersionSnapshotRegistry
from superbiz_agent.memory.schemas import (
    CORE_BLOCK_SPECS,
    CoreMemoryBlock,
)


@dataclass(frozen=True)
class CoreMemoryUpdateResult:
    success: bool
    status: Literal["updated", "unchanged", "rejected"]
    block: CoreMemoryBlock | None = None
    rejection: PolicyDecision | None = None


class CoreMemoryService:
    def __init__(
        self,
        repository: MemoryRepository,
        policy: MemoryWritePolicy,
        snapshots: CoreVersionSnapshotRegistry,
    ) -> None:
        self.repository = repository
        self.policy = policy
        self.snapshots = snapshots

    async def load_blocks(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[CoreMemoryBlock]:
        return await self.repository.ensure_default_core_blocks(tenant_id, user_id, agent_id)

    def max_tokens_for(self, block_key: str) -> int:
        spec = CORE_BLOCK_SPECS.get(block_key)
        return spec.max_tokens if spec else 0

    async def update_block_with_policy(
        self,
        run_context: RunContext,
        block_key: str,
        new_content: str,
        change_reason: str | None = None,
    ) -> CoreMemoryUpdateResult:
        request_context = run_context.request_context
        existing: CoreMemoryBlock | None = None
        if block_key in CORE_BLOCK_SPECS:
            existing = next(
                (
                    block
                    for block in await self.repository.list_active_core_blocks(
                        request_context.tenant_id or "",
                        request_context.user_id or "",
                        request_context.agent_id or "",
                    )
                    if block.block_key == block_key
                ),
                None,
            )
        decision = self.policy.validate_core_update(
            run_context,
            block_key,
            new_content,
            read_only=bool(existing and existing.read_only),
        )
        if not decision.allowed:
            return CoreMemoryUpdateResult(
                success=False,
                status="rejected",
                rejection=decision,
            )
        expected_version = self.snapshots.expected_version(run_context, block_key)
        if expected_version is None:
            return CoreMemoryUpdateResult(
                success=False,
                status="rejected",
                rejection=PolicyDecision.reject(
                    "update_context_missing",
                    "Core memory update requires a trusted version from the current run context.",
                    "Start a new run and retry after Core Memory has been loaded.",
                ),
            )
        write = await self.repository.cas_replace_core_content(
            request_context.tenant_id or "",
            request_context.user_id or "",
            request_context.agent_id or "",
            block_key,
            new_content or "",
            expected_version=expected_version,
        )
        if write.block is not None and write.status in {"updated", "unchanged"}:
            self.snapshots.advance(run_context, block_key, write.block.version)
            return CoreMemoryUpdateResult(
                success=True,
                status=write.status,
                block=write.block,
            )
        rejection_by_status = {
            "read_only": PolicyDecision.reject(
                "read_only_block",
                f"Core memory block '{block_key}' is read-only",
                "Do not attempt to update administrator-managed blocks.",
            ),
            "inactive": PolicyDecision.reject(
                "core_block_inactive",
                "Core memory block is inactive or missing.",
                "Do not retry this update until the block state is repaired.",
            ),
            "conflict": PolicyDecision.reject(
                "update_conflict",
                "Core memory changed after this run loaded it.",
                "Start a new run, review the latest Core Memory, and submit a fresh replacement.",
            ),
        }
        return CoreMemoryUpdateResult(
            success=False,
            status="rejected",
            rejection=rejection_by_status[write.status],
        )

    @staticmethod
    def build_context_block(blocks: list[CoreMemoryBlock]) -> str:
        if not blocks:
            return ""
        lines = ["<core_memory>"]
        for block in blocks:
            lines.append(f"  <block key={quoteattr(block.block_key)}>")
            lines.append(f"    <description>{escape(block.description)}</description>")
            metadata = f"version={block.version} max_tokens={block.max_tokens}"
            if block.read_only:
                metadata += " read_only=true"
            lines.append(f"    <metadata>{escape(metadata)}</metadata>")
            lines.append(f"    <content>{escape(block.content)}</content>")
            lines.append("  </block>")
        lines.append("</core_memory>")
        return "\n".join(lines)
