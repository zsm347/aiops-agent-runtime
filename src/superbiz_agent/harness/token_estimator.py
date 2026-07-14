from __future__ import annotations

import json
import math
from typing import Protocol

from superbiz_agent.model_gateway.base import ModelMessage


class TokenEstimator(Protocol):
    def estimate_text(self, text: str) -> int:
        ...

    def estimate_messages(self, messages: list[ModelMessage]) -> int:
        ...


class ApproxTokenEstimator:
    """Small deterministic estimator used until provider token counters are wired in."""

    message_overhead_tokens = 4
    tool_message_overhead_tokens = 8
    tool_call_overhead_tokens = 12

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        ascii_count = 0
        cjk_count = 0
        other_count = 0
        for char in text:
            codepoint = ord(char)
            if codepoint < 128:
                ascii_count += 1
            elif _is_cjk(codepoint):
                cjk_count += 1
            else:
                other_count += 1
        estimated = (ascii_count / 4.0) + (cjk_count / 1.7) + (other_count / 2.0)
        return max(1, int(math.ceil(estimated)))

    def estimate_messages(self, messages: list[ModelMessage]) -> int:
        total = 0
        for message in messages:
            overhead = self.message_overhead_tokens
            if message.role == "tool":
                overhead += self.tool_message_overhead_tokens
            total += overhead
            total += self.estimate_text(message.role)
            total += self.estimate_text(message.name or "")
            total += self.estimate_text(message.tool_call_id or "")
            total += self.estimate_text(message.content)
            for tool_call in message.tool_calls:
                total += self.tool_call_overhead_tokens
                total += self.estimate_text(tool_call.id)
                total += self.estimate_text(tool_call.name)
                total += self.estimate_text(
                    json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True, default=str)
                )
        return total


def _is_cjk(codepoint: int) -> bool:
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0xF900 <= codepoint <= 0xFAFF
    )
