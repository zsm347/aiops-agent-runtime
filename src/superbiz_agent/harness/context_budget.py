from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextBudget:
    max_input_tokens: int = 32_000
    reserved_output_tokens: int = 4_000
    compaction_trigger_ratio: float = 0.85
    compaction_target_ratio: float = 0.65
    recent_turns_to_keep: int = 4
    recent_keep_ratio: float = 0.70
    tool_result_compress_threshold_tokens: int = 1_000
    content_compression_backend: str = "headroom"
    tool_results_to_keep: int = 3
    compaction_prompt_version: str = "context-compaction-v1"

    @property
    def effective_input_budget(self) -> int:
        return max(1, self.max_input_tokens - max(0, self.reserved_output_tokens))

    @property
    def trigger_tokens(self) -> int:
        return max(1, int(self.effective_input_budget * self.compaction_trigger_ratio))

    @property
    def target_tokens_after_compaction(self) -> int:
        return max(1, int(self.effective_input_budget * self.compaction_target_ratio))

    def should_compact(self, estimated_input_tokens: int) -> bool:
        return estimated_input_tokens > self.trigger_tokens

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "maxInputTokens": self.max_input_tokens,
            "reservedOutputTokens": self.reserved_output_tokens,
            "effectiveInputBudgetTokens": self.effective_input_budget,
            "compactionTriggerRatio": self.compaction_trigger_ratio,
            "compactionTriggerTokens": self.trigger_tokens,
            "compactionTargetRatio": self.compaction_target_ratio,
            "compactionTargetTokens": self.target_tokens_after_compaction,
            "recentTurnsToKeep": self.recent_turns_to_keep,
            "recentKeepRatio": self.recent_keep_ratio,
            "toolResultCompressThresholdTokens": self.tool_result_compress_threshold_tokens,
            "contentCompressionBackend": self.content_compression_backend,
            "toolResultsToKeep": self.tool_results_to_keep,
            "compactionPromptVersion": self.compaction_prompt_version,
        }


def context_budget_from_settings(settings: object) -> ContextBudget:
    return ContextBudget(
        max_input_tokens=int(getattr(settings, "context_max_tokens")),
        reserved_output_tokens=int(getattr(settings, "context_reserved_output_tokens")),
        compaction_trigger_ratio=float(getattr(settings, "context_compaction_trigger_ratio")),
        compaction_target_ratio=float(getattr(settings, "context_compaction_target_ratio")),
        recent_turns_to_keep=int(getattr(settings, "context_recent_turns_to_keep")),
        recent_keep_ratio=float(getattr(settings, "context_recent_keep_ratio")),
        tool_result_compress_threshold_tokens=int(
            getattr(settings, "context_tool_result_compress_threshold_tokens")
        ),
        content_compression_backend=str(getattr(settings, "context_content_compression_backend")),
        tool_results_to_keep=int(getattr(settings, "context_tool_results_to_keep")),
        compaction_prompt_version=str(getattr(settings, "context_compaction_prompt_version")),
    )
