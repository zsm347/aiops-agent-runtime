from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from xml.sax.saxutils import escape, quoteattr

from superbiz_agent.harness.context import RunContext
from superbiz_agent.memory.policy import MemoryWritePolicy, PolicyDecision
from superbiz_agent.memory.schemas import (
    CORE_BLOCK_SPECS,
    DEFAULT_CORE_BLOCK_KEYS,
    CoreMemoryBlock,
)
from superbiz_agent.memory.store import InMemoryMemoryStore, content_hash


@dataclass(frozen=True)
class CoreMemoryUpdateResult:
    success: bool
    status: Literal["updated", "unchanged", "rejected"]
    block: CoreMemoryBlock | None = None
    rejection: PolicyDecision | None = None


class CoreMemoryService:
    def __init__(self, store: InMemoryMemoryStore, policy: MemoryWritePolicy) -> None:
        self.store = store
        self.policy = policy

    def load_blocks(self, tenant_id: str, user_id: str, agent_id: str) -> list[CoreMemoryBlock]:
        existing = {block.block_key: block for block in self.store.load_core_blocks(tenant_id, user_id, agent_id)}
        blocks: list[CoreMemoryBlock] = []
        for block_key in DEFAULT_CORE_BLOCK_KEYS:
            block = existing.get(block_key)
            if block is None:
                block = self.store.initialize_core_block(tenant_id, user_id, agent_id, block_key)
            blocks.append(block)
        return blocks

    def max_tokens_for(self, block_key: str) -> int:
        spec = CORE_BLOCK_SPECS.get(block_key)
        return spec.max_tokens if spec else 0

    def update_block_with_policy(
        self,
        run_context: RunContext,
        block_key: str,
        new_content: str,
        change_reason: str | None = None,
    ) -> CoreMemoryUpdateResult:
        request_context = run_context.request_context
        existing = None
        if block_key in CORE_BLOCK_SPECS:
            existing = next(
                (
                    block
                    for block in self.load_blocks(
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
        if existing and existing.content_hash == content_hash(new_content):
            return CoreMemoryUpdateResult(
                success=True,
                status="unchanged",
                block=existing,
            )
        block = self.store.update_core_content(
            request_context.tenant_id or "",
            request_context.user_id or "",
            request_context.agent_id or "",
            block_key,
            new_content or "",
        )
        if block is None:
            return CoreMemoryUpdateResult(
                success=False,
                status="rejected",
                rejection=PolicyDecision.reject(
                    "update_failed",
                    "Failed to persist core memory update",
                    "Retry the update or check the database connection.",
                ),
            )
        return CoreMemoryUpdateResult(success=True, status="updated", block=block)

    def build_context_block(self, tenant_id: str, user_id: str, agent_id: str) -> str:
        blocks = self.load_blocks(tenant_id, user_id, agent_id)
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
