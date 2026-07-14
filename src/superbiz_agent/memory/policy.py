from __future__ import annotations

import re
from dataclasses import dataclass

from superbiz_agent.harness.context import RunContext
from superbiz_agent.memory.dedup import (
    canonical_content_hash,
    canonicalize_archival_content,
)
from superbiz_agent.memory.schemas import CORE_BLOCK_SPECS, DEFAULT_CORE_BLOCK_KEYS


MAX_ARCHIVAL_CONTENT_CHARS = 4000
MAX_ARCHIVAL_TOPIC_CHARS = 120

SECRET_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?key|password|passwd|pwd|token|bearer)"
        r"\s*[:=]\s*\S+"
    ),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
    re.compile(r"\b(ghp|gho|ghu|ghs|ghr|glpat)_[A-Za-z0-9]{20,}"),
)

RAW_DUMP_PATTERNS = (
    re.compile(r"(?m)^\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.,]?\d{0,3}"),
    re.compile(r"(?m)^\s+at\s+[\w.$]+\([\w.$]+:\d+\)\s*$"),
    re.compile(r"(?m)^[\w_:]+\{[^}]*\}\s+\S+"),
    re.compile(r"^\s*\{.{2000,}\}\s*$", re.DOTALL),
)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    error_type: str | None = None
    message: str | None = None
    suggestion: str | None = None

    @classmethod
    def allow(cls) -> "PolicyDecision":
        return cls(allowed=True)

    @classmethod
    def reject(cls, error_type: str, message: str, suggestion: str | None = None) -> "PolicyDecision":
        return cls(
            allowed=False,
            error_type=error_type,
            message=message,
            suggestion=suggestion,
        )


@dataclass(frozen=True)
class ArchivalValidation:
    decision: PolicyDecision
    content_hash: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision.allowed


class MemoryWritePolicy:
    def __init__(self, *, core_max_tokens: dict[str, int] | None = None) -> None:
        self.core_max_tokens = {
            key: spec.max_tokens for key, spec in CORE_BLOCK_SPECS.items()
        } | (core_max_tokens or {})

    def validate_core_update(
        self,
        run_context: RunContext | None,
        block_key: str | None,
        new_content: str | None,
        *,
        read_only: bool = False,
    ) -> PolicyDecision:
        context_decision = self._validate_context(run_context)
        if not context_decision.allowed:
            return context_decision
        if not block_key or not block_key.strip():
            return PolicyDecision.reject(
                "invalid_block_key",
                "blockKey is required",
                f"Provide one of: {', '.join(DEFAULT_CORE_BLOCK_KEYS)}",
            )
        if block_key not in CORE_BLOCK_SPECS:
            return PolicyDecision.reject(
                "invalid_block_key",
                f"blockKey '{block_key}' is not a supported core memory block",
                f"Use one of: {', '.join(DEFAULT_CORE_BLOCK_KEYS)}",
            )
        if read_only:
            return PolicyDecision.reject(
                "read_only_block",
                f"Core memory block '{block_key}' is read-only",
                "Do not attempt to update administrator-managed blocks.",
            )
        if new_content is None:
            return PolicyDecision.reject(
                "empty_content",
                "newContent must not be null",
                "Provide the full updated block content.",
            )
        max_tokens = self.core_max_tokens.get(block_key, 0)
        approx_tokens = estimate_tokens(new_content)
        if max_tokens > 0 and approx_tokens > max_tokens:
            return PolicyDecision.reject(
                "content_too_long",
                f"newContent is ~{approx_tokens} tokens, exceeds block budget of {max_tokens}",
                "Compress, merge or drop older items and resubmit the full block.",
            )
        sensitive = reject_if_sensitive(new_content)
        if not sensitive.allowed:
            return sensitive
        return reject_if_raw_dump(new_content)

    def validate_archival(
        self,
        run_context: RunContext | None,
        topic: str | None,
        content: str | None,
        evidence_summary: str | None = None,
        *,
        scope_service: str | None = None,
        scope_env: str | None = None,
        tags: list[str] | None = None,
    ) -> ArchivalValidation:
        context_decision = self._validate_context(run_context)
        if not context_decision.allowed:
            return ArchivalValidation(context_decision)
        if not topic or not topic.strip():
            return ArchivalValidation(
                PolicyDecision.reject(
                    "empty_topic",
                    "topic must not be blank",
                    "Use a short, stable topic such as 'service/issue-type'.",
                )
            )
        if len(topic) > MAX_ARCHIVAL_TOPIC_CHARS:
            return ArchivalValidation(
                PolicyDecision.reject(
                    "topic_too_long",
                    f"topic exceeds {MAX_ARCHIVAL_TOPIC_CHARS} characters",
                    "Shorten the topic to a concise slug.",
                )
            )
        if not content or not content.strip():
            return ArchivalValidation(
                PolicyDecision.reject(
                    "empty_content",
                    "content must not be blank",
                    "Summarize the durable lesson in a few sentences.",
                )
            )
        if len(content) > MAX_ARCHIVAL_CONTENT_CHARS:
            return ArchivalValidation(
                PolicyDecision.reject(
                    "content_too_long",
                    f"content exceeds {MAX_ARCHIVAL_CONTENT_CHARS} characters",
                    "Summarize the memory; do not paste raw logs or tool output.",
                )
            )
        for text in (topic, content, evidence_summary, scope_service, scope_env, *(tags or [])):
            sensitive = reject_if_sensitive(text)
            if not sensitive.allowed:
                return ArchivalValidation(sensitive)
        if not canonicalize_archival_content(content):
            return ArchivalValidation(
                PolicyDecision.reject(
                    "empty_content",
                    (
                        "content must contain meaningful text and cannot consist only "
                        "of whitespace or sentence-ending punctuation"
                    ),
                    "Provide a concise durable lesson with actual content.",
                )
            )
        raw_dump = reject_if_raw_dump(content)
        if not raw_dump.allowed:
            return ArchivalValidation(raw_dump)
        return ArchivalValidation(
            PolicyDecision.allow(),
            canonical_content_hash(content),
        )

    @staticmethod
    def _validate_context(run_context: RunContext | None) -> PolicyDecision:
        if run_context is None:
            return PolicyDecision.reject(
                "missing_context",
                "Runtime context is required",
                "Memory writes must happen within an agent run.",
            )
        request_context = run_context.request_context
        if not _present(request_context.tenant_id):
            return PolicyDecision.reject("missing_tenant", "tenantId is required")
        if not _present(request_context.user_id):
            return PolicyDecision.reject("missing_user", "userId is required")
        if not _present(request_context.agent_id):
            return PolicyDecision.reject("missing_agent", "agentId is required")
        return PolicyDecision.allow()


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def reject_if_sensitive(text: str | None) -> PolicyDecision:
    if not text:
        return PolicyDecision.allow()
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return PolicyDecision.reject(
                "sensitive_data",
                "Content appears to contain secrets or credentials",
                "Remove secrets, tokens, passwords, API keys and private keys before writing memory.",
            )
    return PolicyDecision.allow()


def reject_if_raw_dump(text: str | None) -> PolicyDecision:
    if not text:
        return PolicyDecision.allow()
    matches = sum(1 for pattern in RAW_DUMP_PATTERNS if pattern.search(text))
    if matches >= 2:
        return PolicyDecision.reject(
            "raw_log_not_allowed",
            "Content appears to be raw logs, alerts, metrics or a large tool output dump",
            "Summarize the verified lesson in your own words instead of pasting raw output.",
        )
    return PolicyDecision.allow()


def _present(value: str | None) -> bool:
    return value is not None and bool(value.strip())
